"""Vertex AI (Gemini) image localizer: decide whether a PDF image needs cultural
adaptation for a Bangladeshi audience, and if so, edit it in place.

Two model calls per image:
  1. classify_image  — cheap multimodal Gemini call, structured JSON decision.
  2. localize_image  — image-editing model that returns an edited image.

Both fail safe: on any error the caller keeps the original image, so the PDF is
never corrupted (mirrors translator._translate_chunk)."""

import json
import logging
import re
import time

from google.genai import types

from vertex_client import IMAGE_TIMEOUT_MS, generate_content

logger = logging.getLogger(__name__)

# Use available models only in this Vertex AI project
CLASSIFY_MODEL = "gemini-3-flash-preview"  # For classification and text translation
TEXT_TRANSLATE_MODEL = "gemini-3-flash-preview"  # For translating text to Bangla

# The image model. One model, no fallback chain — set by the user 2026-08-04.
#
# What the chain used to be, and why it was the wrong shape for this job. Measured on the Heart
# Failure manual's cover figure (a person holding a blank placard), placard position after
# regeneration vs. before — full table in image_regen.EDIT_MODELS:
#
#   gemini-3.1-flash-lite-image   IoU 0.95   aspect drift 0.04%   ~9s
#   gemini-2.5-flash-image        IoU 0.72   aspect drift 2.95%   ~14s   (snaps to a 2:3 bucket)
#   gemini-3.1-flash-image        IoU 0.68   aspect drift 0.04%   ~19s
#   gemini-3-pro-image            IoU 0.60   aspect drift 1.08%   ~51s   (reinterprets the most)
#
# The chain was ordered by *framing fidelity*, because a pinned edit restores text onto blank
# surfaces at positions measured BEFORE the edit, so a model that shifts a placard leaves that
# text hanging off it. The consequence nobody had looked at: the model tried FIRST is the one
# measured to change the least. On a redraw it returns a competent retouch, the loop is
# satisfied, and no stronger model is ever called — which is exactly the "you only changed the
# character" complaint, and no amount of prompt wording could reach it.
#
# gemini-3.1-flash-image reinterprets substantially more than the lite model (IoU 0.68 vs 0.95)
# while holding the aspect ratio as tightly as it does (0.04%), which matters because the
# picture is printed into a fixed rectangle. ~19s per picture rather than ~9s. It costs no extra
# QUOTA: that is per-minute and shared across every image model, so a chain was never a quota
# escape — it only ever helped when a model refused a particular picture, which is the one thing
# a single model gives up. A refusal now means that picture ships un-localized, recorded in the
# audit as edit_failed with edit_error="refused".
EDIT_MODEL = "gemini-3.1-flash-image"
EDIT_MODELS = [EDIT_MODEL]
# Attempts per model before moving to the next one (transient 429/503 backoff is handled inside
# vertex_client.generate_content, so a couple of attempts per model is plenty).
EDIT_ATTEMPTS_PER_MODEL = 2

# Quality over speed: an image that failed only because the project was out of quota must not
# ship untouched. The image quota is per-MINUTE and shared by every image model, so waiting
# out a minute is the only thing that clears it — the model chain cannot. Two extra passes
# costs at most ~2.5 minutes on a picture that would otherwise be lost.
EDIT_QUOTA_RETRY_PASSES = 2
EDIT_QUOTA_COOLDOWN_SEC = 70.0

MAX_ATTEMPTS = 5

# The four culturally-specific categories the user asked to localize. Anything
# that fits none of these (logos, charts, diagrams, icons, decorative graphics,
# UI screenshots, or already-Bangladeshi content) is left untouched.
CATEGORIES = ["people_attire", "scenes_settings", "food_objects", "signage_text"]

CLASSIFY_SYSTEM_PROMPT = """You decide whether an image inside a document should be \
adapted for a Bangladeshi audience, so the document feels local to Bangladeshi readers.

Return needs_localization=true if the image depicts culturally-adaptable content in one \
or more of these categories:
- people_attire: ANY people or faces (photo, illustration, cartoon, or icon of a person). \
Their appearance and clothing can always be made Bangladeshi (saree, salwar kameez, \
panjabi, hijab), so an image containing a person should almost always be localized.
- scenes_settings: streets, buildings, landscapes, rooms, or environments (including \
diagrams, charts showing Western settings, or infographics with non-Bangladeshi visual context)
- food_objects: meals, produce, or everyday cultural objects (including food plates, \
beverage photos, eating utensils, or food-related diagrams/charts)
- signage_text: readable signs, labels, or text baked into the image (render in Bangla)

Be decisive, not cautious: if any person is visible, return true even for a simple or \
stylized illustration. A generic "Western/international" look is exactly what should be \
localized — do not keep an image just because it looks neutral. Include diagrams and \
infographics if they show Western contexts, food, or objects that should be made local. \
Even text-heavy images of food, scenes, or objects should be localized — the text will be \
handled explicitly during image editing.

A cartoon, line drawing, sketch or simplified illustration of a REAL thing is not \
"abstract" — it is a picture of that thing, and it counts. A cartoon car is a car (a \
vehicle: an object). A cartoon person is a person. A hand-drawn house is a building. A \
simple sketch of a plate of food is food. Style has nothing to do with it: only a diagram \
of a pure concept — a flowchart of ideas, an arrow, a graph of quantities, a gradient or \
an ornamental shape — is abstract. If you can name the real-world thing the drawing shows, \
and that thing is a person, a place, a vehicle, a building, a garment, a utensil, a piece \
of furniture, a household object or food, return needs_localization=true.

Return needs_localization=false ONLY for:
- Pure medical/clinical images: x-rays, CT scans, anatomical diagrams, clinical charts, \
and medical reference images where the image's primary purpose is technical/diagnostic.
- Functional technical elements that must remain intact: QR codes, barcodes, UI chrome, \
buttons, decorative rules, arrows, ornamental shapes, colour gradients, and flowcharts or \
graphs of pure abstract concepts. These are things with no real-world subject at all — do \
not put a drawing of a real object in this group because it is drawn simply.
- Images already looking authentically Bangladeshi.

List only the categories that apply.

FIRST, before anything else, answer is_logo. A logo is the identity of a real organisation \
and is never ours to change — not its drawing, not its colours, and above all not its \
wording.

Answer is_logo=true whenever a mark is present anywhere in the image, then answer \
logo_fills_image to say WHICH of the two situations it is:
- logo_fills_image=true — the image IS the mark. The mark and its immediate lockup are \
essentially the whole picture; there is nothing else in the frame but the mark and its \
background. This image will be left exactly as printed.
- logo_fills_image=false — the image CONTAINS a mark. It is a photograph, illustration, \
diagram or chart with an organisation's mark somewhere in it, typically in a corner or along \
an edge, and the rest of the frame is a real subject: people, food, a place, a chart. The \
mark is found and protected separately by its own detector — its pixels stay as printed and \
its wording is never translated — while the rest of the picture is localized normally.
Getting this wrong in the "fills" direction is expensive: a whole photograph of a meal was \
left in English because a food agency's crest sat in its top corner. If the frame holds a \
real subject as well as the mark, answer false.

Return is_logo=true for:
- Any logo, wordmark, lettermark, emblem, crest, seal, coat of arms, badge or roundel.
- Any brand or product mark, and any charity, hospital, trust, university, government, \
ministry, NHS, WHO or other institutional mark.
- A name or initials set as an organisation's identity — a stylised wordmark, a name locked \
up with a symbol, a masthead — even when it is only lettering.
- A strapline or tagline printed as part of such a mark.
- Copyright lines, registration numbers, ISBNs and publisher imprints.
- Any of the above even when it also contains people, a building, a plant or a landscape: a \
crest with a lion in it is a crest, not a picture of a lion.
When is_logo=true AND logo_fills_image=true, needs_localization MUST be false, and the \
mark's text must NOT be translated. When is_logo=true but logo_fills_image=false, judge \
needs_localization on the rest of the picture as usual — the mark itself is protected \
elsewhere. If you are unsure whether a mark is a logo, answer is_logo=true; if you are \
unsure whether it fills the frame, answer logo_fills_image=false, because a picture wrongly \
called a whole logo is deleted from the localization entirely while a mark inside a picture \
is still protected.

Separately, judge information_role — what the picture is FOR. This is not the same \
question as whether it can be localized, and you must answer it independently:
- "referential": the specific thing shown IS a datum the page states in words. Redrawing it \
as something else would make the page factually WRONG, not merely less local. This is: a \
drink or a food pictured to define a measure, a unit or a dose ("1.5 units", "one portion = \
80g"); a labelled specimen, product, tablet or piece of equipment the reader is meant to \
recognise; one tile of a chart, key, grid or comparison series whose tiles are being \
contrasted with each other; and any picture printed beside a number, percentage or quantity \
that describes what is in the picture.
- "decorative": everything else, INCLUDING ordinary pictures of food and meals. A plate of \
food illustrating what balanced eating looks like, a family at a meal, someone talking to a \
nurse, a person walking, a figure holding a sign. These are decorative because the document \
states nothing factual about the particular dish or the particular person shown. A picture \
is not referential merely because it shows food, or because the page it sits on is about \
health.
Worked examples: a photograph of a plate divided into food groups, printed to show what a \
balanced diet looks like -> decorative (the groups are what matter, and the redraw is \
separately required to keep the same food groups and the same portions). One card in a row \
of eight, each showing a drink beside the number of alcohol units in it -> referential.
When the two readings are genuinely both arguable, answer "referential" — a picture redrawn \
when it should not have been is a factual error in the document, while one left alone is \
merely un-localized."""

# Three ways to regenerate a picture, chosen per image by image_processor._regeneration_mode:
#
#   "context" — the page's own words go to the model with the picture, so what comes back
#   still illustrates the paragraph it sits beside. A figure printed next to "walk for 20
#   minutes every day" comes back walking rather than sitting, and a plate beside a section
#   on portion sizes keeps being a plate.
#
#   "simple"  — a straight cultural swap with no page text. Used where there is no real
#   context to give (a margin icon, a cover ornament, a picture on an otherwise blank
#   divider): a context clause assembled from three stray words is worse than none, because
#   the model reads whatever it is handed as a brief and draws to it.
#
#   "reimagine" — the picture is DRAWN AGAIN as a Bangladeshi scene rather than retouched.
#   The other two modes pin every element to its original position, because text read off
#   the original is painted back into those same boxes afterwards; an edit model handed
#   "nothing moves and nothing resizes" complies the cheapest way it can, which is to change
#   the faces and the clothes and leave the Western room, street, furniture and props exactly
#   as drawn. That is the "only the characters changed" result. Where the picture carries no
#   baked text there is nothing to line up with, so the frame can be freed: same message,
#   same aspect ratio, same style and palette family, but the scene itself is composed from
#   what the subject actually looks like in Bangladesh. See image_processor._regeneration_mode
#   for the conditions — it is gated, not the default, because a recomposed picture and a
#   fixed text overlay cannot both be right.
LOCALIZE_MODES = ("context", "simple", "reimagine")

# Concrete visual vocabulary, shared by every mode.
#
# "Make it Bangladeshi" is not an instruction a model can execute — asked for it and nothing
# else, it does the one substitution it is certain of (skin, hair, a saree) and leaves the
# rest of the frame as it found it. Naming the props is what actually moves a picture: a
# model that has been told "rickshaw, tin roof, ceiling fan, steel plate, cha in a glass cup"
# draws a Bangladeshi room, while one told "adapt the setting" draws a Western room with a
# Bangladeshi person in it. Same reason the palette is given as measured hex values rather
# than as "match the original" — see image_processor._palette_summary.
BANGLADESHI_VOCABULARY = """WHAT "BANGLADESHI" LOOKS LIKE — draw from this, do not stop at \
skin and clothing:
  - PEOPLE: Bengali faces — rounded jaw, broad nose, dark brown eyes, thick black hair; warm \
brown skin across a real range from fair-wheatish to deep brown. Not East Asian, not Arab, \
not a tanned European.
  - DRESS: women in a cotton saree worn with the anchol over the shoulder, or a salwar kameez \
with the orna across the chest, many with a hijab; men in a panjabi with pyjama, a shirt with \
full trousers, or a lungi with a shirt or genji, some in a topi; older men often with a \
beard; children in school uniform, girls with plaited hair and ribbons. Sandals or chappals, \
not trainers, indoors nothing on the feet.
  - HOMES: brick or plastered walls painted pale green, blue or cream; a corrugated tin roof \
or a flat concrete one; cement or red-oxide floors; a ceiling fan; a mosquito net over a \
wooden khat; plastic moulded chairs, a wooden almirah, a calendar and a wall clock; a \
jaynamaz; steel or melamine crockery; a jug and glass on a tray.
  - STREETS AND VILLAGES: cycle rickshaws with painted hoods, green CNG auto-rickshaws, \
crowded buses, vans and hawker carts; a tea stall with a kettle and small glass cups; shops \
behind corrugated shutters; tangled overhead cables; brick-paved lanes; a mosque minaret; \
a pond (pukur) with steps down to it, paddy fields, banana, coconut and betel-nut palms, \
bamboo, a river with a wooden nouka, a monsoon sky.
  - HEALTH SETTINGS: a community clinic or upazila health complex — pale green or white \
painted walls, a metal bed with a plain sheet, a curtain rail, a wooden desk; a doctor in a \
white coat over a saree or a panjabi; a health worker or paramedic with a register; not a \
Western hospital corridor.
  - FOOD: bhat, dal, machher jhol with rui or ilish, shobji bhaji, ruti, khichuri, doi, muri; \
cha in a small glass cup; seasonal fruit — aam, kathal, kola, peyara, boroi; served on steel \
or melamine plates and eaten with the right hand, or with a steel spoon.
Draw Bangladesh specifically, not a generic "South Asian" or "Middle Eastern" stand-in: no \
desert, no pagoda, no Gulf skyline, no Western suburban kitchen or lawn.

NOT BANGLADESHI — if any of these is in the original it does NOT survive into your picture. \
Replace it with whatever fills the same role in Bangladesh, or leave it out:
  - CLOTHING AND KIT: hi-vis or reflective safety jackets, tabards and vests; work gloves of \
any colour, mittens, latex gloves outside a clinical scene; hard hats, bicycle helmets, \
beanies, baseball caps; winter coats, anoraks, fleeces, hoodies, scarves, blazers, ties, \
suits, jeans; trainers, boots, wellingtons, high heels; Western-cut backpacks and handbags; \
sunglasses; lanyards and clipboards.
  - HOUSEHOLD AND STREET: fitted kitchens with wall units and worktops, ovens, kettles, \
toasters, microwaves, dishwashers, fridges with magnets; sofas with scatter cushions, \
carpets, fireplaces, radiators, sash or double-glazed windows, net curtains, wallpaper; \
wheelie bins, letterboxes, garden fences, mown lawns, hedges, kerbed pavements, zebra \
crossings, Western road signs, cars, Western buses, supermarket trolleys.
  - FOOD AND TABLE: a knife and fork laid beside a plate, dinner plates on placemats, mugs of \
milky tea, bottled water, breakfast cereal, sandwiches, toast, a roast dinner.
  - WEATHER AND LAND: snow, autumn leaves, bare deciduous trees, a grey European sky, pine \
forest, snow-capped mountains, a Western high street.
Safety and cold-weather kit is the most persistent of these: a person is NOT localized while \
they are still wearing the original's gloves, jacket, boots, helmet or hi-vis. Go through \
every garment and every object a person is wearing, holding or standing next to, one at a \
time, and ask whether a Bangladeshi in this scene would have it. If not, it goes."""

CONTEXT_CLAUSE = """CONTEXT — the page this picture is printed on says:
"{context}"
Use it to keep the picture's MEANING intact: the same activity, the same kind of objects, \
the same number of people doing the same thing. It tells you what the picture is FOR — it \
is not a list of things to add, and not one word of it may be written into the image.
"""

EDIT_INSTRUCTION = """Edit this image to reflect Bangladeshi culture and context. Keep the \
EXACT composition, framing, camera angle, and same aspect ratio and dimensions.
{context_clause}CRITICAL — NOTHING MOVES AND NOTHING RESIZES:
  - Every element stays at exactly the same position and exactly the same size as in the \
original. Do not shift, rotate, rescale, crop, re-centre, or re-compose anything.
  - This matters most for blank surfaces: a sign, board, placard, card or panel must keep \
its edges, its corners and its size to the pixel. Real text is printed onto those surfaces \
afterwards at fixed positions, so a board that moves or shrinks leaves that text hanging \
off it and overlapping the artwork.
  - Do not zoom in or out, and do not change how much of the subject is visible or how much \
empty space surrounds it.
  - Return the picture at the SAME aspect ratio you were given. Do not pad it, do not crop \
it, and do not fit it into a square or a 2:3 frame.
  - If you cannot keep a board, card or placard exactly where it is, leave that part of the \
picture unchanged rather than moving it. A blank surface that has shifted is worse than one \
that was never adapted: real text is printed onto it afterwards at fixed positions.
{locks}
CRITICAL — MATCH THE ORIGINAL'S COLOURS:
  - This picture is printed inside a document, surrounded by the page it sits on. It must \
still look like it belongs to that page, so the palette is not yours to change.
  - Reproduce the SAME colour palette as the original: the same hues, the same lightness, \
the same level of saturation, the same overall tone.{palette}
  - The background must stay the EXACT same colour as in the original — if it is plain, \
keep that plain colour; if it is white, keep it white. Never replace a plain background \
with a scene, gradient, texture, or a different colour.
  - Do NOT boost saturation, warm the image up, add new accent colours, or restyle it. \
A recoloured picture is a failure even if it looks nice on its own.
  - If the original is a flat line drawing or a limited two- or three-colour illustration, \
keep exactly that style — do not turn it into a photo, a painting, or a shaded 3D render.{style}
  - It must look like something PRINTED in a booklet, not something generated: no soft glow, \
no bloom, no vignette, no gradient mesh, no drop shadows, no glossy or plastic highlights, no \
depth-of-field blur, no sparkles or bokeh, no over-rendered 3D lighting.
- Clarity: the output must be CLEARER than the original — cleaner and steadier lines, \
sharper edges, better-resolved detail, no blur or compression artifacts. Clarity comes \
from draughtsmanship, not from stronger colour: sharpen the drawing, not the palette.
CRITICAL — PEOPLE ARE REDRAWN, NOT RE-DRESSED:
  - Every visible person must BE Bangladeshi, not a Western person wearing Bangladeshi \
clothes. Changing only the clothing is the most common failure and is not acceptable.
  - Redraw the person: face shape, nose, lips, eyes, brow and jaw as a Bangladeshi person's; \
warm brown South Asian skin, and black or dark brown hair with South Asian hair texture. No \
blond, red, ginger or light-brown hair, no blue or green eyes, no pale or pink complexion, \
no Western facial structure.
  - Facial hair, hairstyle and any head covering should read as they would in Bangladesh for \
that person's age, gender and role.
  - THEN dress them: saree, salwar kameez, panjabi, kurta, hijab, lungi, or ordinary modern \
Bangladeshi clothing as fits their age, gender and context.
  - Keep the same number of people, the same poses, gestures, expressions and positions, at \
the same size and in the original's drawing style. It is the person who changes, not the \
picture's composition. The one exception is the behaviour rules below: where they require a \
change, adjust only the limbs, clothing or spacing involved and leave the composition, the \
count and everyone's place in the frame untouched.
CRITICAL — BEHAVIOUR MUST FIT BANGLADESH, NOT ONLY APPEARANCE:
  - A Bangladeshi-looking person acting out a Western scene is the same failure as a Western \
face in a saree. What the people are doing, what they are wearing and how they touch each \
other must read as ordinary and respectable in Bangladesh.
  - MODESTY — everyone is covered. Women wear a saree, a salwar kameez with the orna over the \
chest, or other loose full-length clothing: shoulders, arms, chest, midriff and legs covered. \
Men wear a shirt with full trousers, pyjama or lungi, and are not bare-chested. No shorts, \
vests, sleeveless or low-cut tops, clinging fits, short skirts, swimwear or gym-wear, and no \
bare legs — this holds during exercise, sport, swimming, at the beach and at home. Draw the \
SAME activity in modest, covered clothing rather than dropping the activity.
  - CONTACT BETWEEN MEN AND WOMEN — none. No hugging, kissing, cheek-kissing, hand-holding, \
an arm around a shoulder or waist, sitting in a lap, or leaning on one another. A husband and \
wife, a carer and a patient, a doctor and a patient stand or sit side by side at a respectful \
distance. Contact within the same gender (a hand on a friend's shoulder) and a parent holding \
their own young child are normal and stay.
  - MANNERS — eat, give and receive with the right hand; feet stay off tables, chairs and \
desks and soles are not turned towards anyone; people greet with salam or a nod, not a kiss \
or an embrace; older people are shown being deferred to.
  - SETTINGS THAT CARRY BEHAVIOUR — a pub, bar, nightclub, dance floor, sunbathing or beach \
scene becomes the Bangladeshi setting that serves the same purpose (a tea stall, a home \
sitting room, a park or riverside walk, a community hall), with the same activity and the \
same number of people. No dogs indoors, on a lap or on furniture.
  - This never overrides the picture's health message: if the booklet is deliberately showing \
a habit to avoid, it still shows it. Adapt how people behave, not what the page is teaching.
CRITICAL — THE WHOLE FRAME IS LOCALIZED, NOT ONLY THE PEOPLE:
  - Changing the faces and the clothing and leaving the room, the street, the furniture, the \
crockery and the props exactly as they were drawn is the single most common failure of this \
task, and the result is rejected. A Bangladeshi family in a Western kitchen is not localized.
  - Every object in the frame that a Bangladeshi household or street would not contain is \
replaced by the thing that fills the same role there — in the same position, at the same \
size, in the original's palette and style. Walls, floors, roofing, windows, doors, furniture, \
crockery, utensils, appliances, vehicles, shopfronts, signage shapes, trees and plants are \
all in scope, not just the people.
  - Work through the frame element by element and ask of each one: is this what it would look \
like in Bangladesh? Anything that would not be there is redrawn.
  - THIS INCLUDES WHAT PEOPLE WEAR AND HOLD. Gloves, hi-vis jackets, tabards, helmets, hats, \
coats, scarves, boots, trainers, bags and the objects in their hands belong to the original's \
world, not to its message. Each one is replaced by what a Bangladeshi doing this would have — \
drawn at the same place and the same size, so the composition is untouched. Keeping the \
original's gloves or jacket on a Bangladeshi face is the commonest way this task is failed.
CRITICAL — FOOD RULES. All three apply, in this order:
  1. HALAL ONLY. Never depict pork, ham, bacon, lard, alcohol, beer, wine, or a wine glass. \
If the original shows one, replace it with a halal food filling the same role.
  2. KEEP THE NUTRITIONAL MEANING. This picture is printed in a health booklet, where a food \
is very often shown to represent a food group, a portion size, a measure or a dose. The \
replacement MUST be in the SAME food group and show the SAME portion: oily fish -> ilish or \
rui (never dal); wholegrain -> lal chal or atta ruti (never white rice); leafy vegetable -> \
lal shak or palong shak; pulse -> dal; dairy -> doi or milk; fruit -> a fruit. Never swap \
across food groups, and never change how much food is shown or how many items are on the plate.
  3. MAKE IT BANGLADESHI. Subject to rules 1 and 2, replace Western dishes with everyday \
Bangladeshi food — bhat, dal, machher jhol, shobji, cha — served on Bangladeshi plates and \
eaten with Bangladeshi utensils.
- Text & signs: {text_instruction}
CRITICAL — NO TEXT:
  - Do NOT draw, write, render, or hallucinate ANY text, letters, words, numbers, or symbols.
  - Every sign, placard, board, label, poster, or text surface must be reproduced BLANK and CLEAN \
(empty paper/board of the same shape, color, and material) — no glyphs of any kind.
  - Text is added back separately after this step, so leaving it out is required, not a mistake.
CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate, or invent any logo, wordmark, emblem, crest, \
badge, or institutional mark. Never substitute a different organisation's mark for the one there.
  - Leave the area a mark occupies as clean, empty background of the surrounding colour. The \
original marks are stamped back on top afterwards, unchanged.
{vocabulary}
Do not add or remove objects, change the layout, or add new elements beyond cultural \
adaptation. Keep all text surfaces present but empty. Focus especially on: {focus}."""

# The compact prompt used in "simple" mode. Not a trimmed copy of the one above for its own
# sake: a small picture — a margin icon, one tile of a row — given thirty lines of art
# direction comes back elaborated, with detail invented to satisfy clauses that were never
# about it. The rules that actually matter at that size are the palette, the blank text
# surfaces, and the framing.
SIMPLE_EDIT_INSTRUCTION = """Redraw this image so it depicts Bangladeshi people, clothing, \
food, and surroundings instead of Western ones.
- Keep the EXACT composition, framing, aspect ratio and dimensions. Same number of subjects, \
same poses, same layout.
- NOTHING moves and NOTHING resizes: every element keeps its exact position and size. Blank \
signs, boards and panels keep their edges to the pixel — text is printed onto them afterwards \
at fixed positions, so one that moves or shrinks leaves that text overlapping the artwork.
- Keep the SAME colour palette, the same style, and the same background colour as the \
original.{palette} Do not brighten, restyle, or turn a flat drawing into a photo.{style}
- Every person must BE Bangladeshi, not a Western person in Bangladeshi clothes — redrawing \
only the outfit is a failure. Redraw the face itself: South Asian face shape and features, \
warm brown skin, black or dark brown hair. No blond or light hair, no pale complexion, no \
Western facial structure. Then dress them (saree, salwar kameez, panjabi, kurta, hijab, \
lungi) to suit their age and role. Same poses, same positions, same count.
- Behaviour must fit Bangladesh too, not just faces. Everyone is modestly covered — no \
shorts, sleeveless or low-cut tops, tight fits, swimwear, gym-wear or bare legs, not even \
while exercising or swimming — and a man and a woman are never shown touching: no hugging, \
kissing, hand-holding or an arm round a shoulder; they stand side by side instead. Same \
gender contact and a parent holding their own child are fine. Change only the clothing and \
the touching — the activity, the poses and the number of people stay as they are.
- Food becomes Bangladeshi and halal — never pork, alcohol, beer or wine — and stays in the \
SAME food group and the SAME portion as the original (oily fish -> ilish or rui, wholegrain \
-> lal chal or atta ruti, pulse -> dal, dairy -> doi), served in Bangladeshi dishes.
- Draw NO text of any kind. Every sign, label or lettered surface comes back blank and clean \
— the text is restored separately afterwards.
{locks}
- Draw NO logo, wordmark, emblem, crest or institutional mark, and never invent or substitute \
one. Leave its area clean and empty; the original mark is stamped back afterwards, unchanged.
- The whole frame is localized, not only the people. Changing the faces and the clothes while \
the room, the street, the furniture, the crockery and the props stay Western is a failure. \
Every object is replaced by what fills the same role in Bangladesh — same position, same \
size, same style: brick or plastered walls and a tin roof, cement floors, a ceiling fan, \
plastic chairs and a wooden khat, steel or melamine plates, cha in a small glass cup, bhat \
and dal and machher jhol, cycle rickshaws and CNG auto-rickshaws, tea stalls, paddy fields, \
banana and coconut palms, a pond. Bangladesh specifically — not a generic South Asian, Middle \
Eastern or Western stand-in.
- Anything Western that a Bangladeshi would not have goes, including what people are WEARING \
and HOLDING: no hi-vis jackets, work gloves, mittens, helmets, hard hats, beanies, coats, \
fleeces, scarves, jeans, trainers or boots; no fitted kitchen, sofa, carpet, radiator, wheelie \
bin, lawn, car or knife and fork; no snow and no bare winter trees. A person still wearing the \
original's gloves or jacket has not been localized.
Focus especially on: {focus}."""

# "reimagine" mode. The picture is drawn again from its meaning rather than edited, so the
# clause that carries the page's words is worded differently too: in the other two modes the
# context is a fence ("keep the meaning intact"), and here it is the brief. It is still not a
# licence to write any of those words into the image.
REIMAGINE_CONTEXT_CLAUSE = """THE BRIEF — this picture is printed on a page that says:
"{context}"
The picture exists to illustrate that. Everything you draw must serve it: the same subject, \
the same activity, the same point being made to the reader. This is what the picture is FOR, \
not a list of objects to include, and not one word of it may appear as text in the image.
"""

# Reserved rectangles: the parts of a picture the page's own printed text sits over.
#
# Those captions are real PDF text at fixed page coordinates and cannot be moved, so whatever
# surface they are printed on — a placard, a panel, a patch of flat field — has to come back in
# exactly the same place, the same size, and blank. Everything else in the picture is free,
# which is the whole point: without locks the only safe instruction is "nothing moves", and
# that is what reduces a redraw to a change of faces.
#
# Assembled by image_processor._lock_summary, which measures the rects and their colours. Three
# layers per rect, in decreasing order of how far they can be trusted: prose position, then
# percentages, then box_2d on the 0-1000 grid. box_2d is the convention the model answers OCR
# in — that is evidence about *reading*, not about drawing — so it is the third layer, and the
# actual guarantee is the pixel check in image_processor._locks_kept. A marker drawn into the
# input image would be simpler and is not an option: the model reproduces markers faithfully.
# NEVER call a reserved rectangle "blank".
#
# The first version of this clause said each one "must come back completely BLANK: flat, empty,
# unmarked surface". The model read "blank surface" as a THING TO DRAW and painted a blank white
# card, with a border, into the middle of a man's jumper — obeying the instruction exactly as
# written. A reserved rectangle is not an object to render; it is a piece of the picture that
# must survive untouched. So the wording says *continue what is already there*, and says
# outright that adding a card, panel or sign is the failure.
LOCK_CLAUSE_HEAD = """RESERVED AREAS — {n} rectangle(s) of this picture are already spoken \
for. The booklet's own printed text is laid over them on the page, at fixed coordinates that \
cannot move, so whatever surface is at each one has to survive your redraw: same place, same \
size, same shape, same colour, still smooth and unbroken.
{items}
These are NOT things to draw. Do not add a card, a panel, a sign, a placard, a box, a frame, a \
label or a patch of flat colour at any of them — there is already a surface at that spot and \
your job is simply to let it continue, exactly as it is, while you redraw everything around \
it. Equally, do not move, resize, rotate, tilt, re-shape or re-colour it, and do not bring a \
hand, a limb, an object, a shadow, an outline, a texture or any lettering across it. If the \
picture you want to draw would put artwork there, put that artwork elsewhere in the frame. A \
reserved area that has moved, been covered, or had something new drawn into it makes the \
booklet's printed text unreadable, and the whole picture is thrown away.

You may redraw everything else in this picture freely — that freedom is the point.
"""
LOCK_ITEM = """  {i}. the {where} of the frame — from {x0:.0%} to {x1:.0%} across and {y0:.0%} \
to {y1:.0%} down, about {w:.0%} wide and {h:.0%} tall (box_2d [{by0},{bx0},{by1},{bx1}] on a \
0-1000 grid). It is smooth {colour} there now; leave it smooth {colour}, unbroken and \
unmarked, and add nothing to it."""
# The compact form for SIMPLE_EDIT_INSTRUCTION, whose brevity is deliberate — see the comment
# above that prompt.
LOCK_ITEM_COMPACT = """  {i}. from {x0:.0%} to {x1:.0%} across and {y0:.0%} to {y1:.0%} down \
({where}) — leave the smooth {colour} that is there, and add nothing."""
LOCK_CLAUSE_COMPACT = """RESERVED — the page's printed text is laid over {n} rectangle(s) of \
this picture, so the surface already at each one must survive unchanged: same place, same \
size, same colour, unbroken. Do NOT draw a card, panel, sign or box there — there is a surface \
there already and it only has to continue.
{items}
Everything else in the picture is yours to redraw.
"""


# Written as a *drawing* brief, not an editing one. The edit prompts above are built around a
# text overlay that is painted back at coordinates measured on the original, so they spend
# most of their length forbidding movement — and a model told to move nothing satisfies the
# cultural instruction with the cheapest possible change, the faces and the clothes. This
# prompt is used only where the picture carries no baked text (image_processor gates it), so
# the frame is free and the instruction can ask for the thing that was actually wanted: the
# scene as it exists in Bangladesh. What it still may not change is what makes the picture fit
# the printed page — aspect ratio, drawing style, palette — and what makes it true: the health
# message, the food group, the number of people doing the thing.
REIMAGINE_INSTRUCTION = """Draw this picture again as it would be drawn for a health booklet \
published in Bangladesh, for Bangladeshi readers.
{context_clause}{locks}
CRITICAL — THIS IS A REDRAW, NOT A RETOUCH:
  - Do not trace the original and swap the faces. Look at what the picture is showing, then \
draw that same thing as it actually happens in Bangladesh — the people, and also the place \
they are in, what they are wearing, what they are holding, what is behind them and what is \
under their feet.
  - A picture that has kept its Western room, street, furniture, crockery, vehicles or \
landscape and changed only the skin, hair and clothing has FAILED this task. That is the \
specific outcome this instruction exists to prevent.
  - You may recompose within the frame: move, resize, add or drop background and secondary \
elements, change the setting, change the props, change the camera distance, so long as the \
result still says exactly what the original said.
  - EVERYTHING A PERSON IS WEARING OR HOLDING IS DRAWN AGAIN FROM SCRATCH, NOT KEPT. Gloves, \
jackets, tabards, hats, helmets, boots, bags, tools and whatever is in their hands belong to \
the original's world, not to its message. Draw the person again from the skin outwards and ask \
what a Bangladeshi doing this would actually be wearing and holding. Keeping even one garment \
or one prop from the original — a pair of gloves, a hi-vis jacket, a helmet — is the tell that \
this was a retouch and not a redraw, and the picture is rejected for it.
  - THE PEOPLE MAY BE POSED DIFFERENTLY. They may stand differently, face a different way, be \
seen from a different angle, hold the thing differently, gesture differently, sit where they \
stood, and be placed differently in the frame. The original's pose is one draughtsman's \
choice, not part of what the picture teaches: if a Bangladeshi doing this would stand, hold or \
carry differently, draw that. The same silhouette with a new face in it is precisely the \
failure this instruction exists to prevent.
  - A PLAIN BACKGROUND IS NOT A SCENE TO FILL IN. If the original's background is a flat, \
plain field, that field stays exactly as it is — the same colour, flat and unbroken to the \
edges. Do not fill it with a room, a street, a landscape or a sky. On a picture like that, \
localize by what the people ARE, wear, hold, stand on and are surrounded by — a low wooden \
stool, a gamcha over a shoulder, a steel jug and glass, a woven pati, a handful of shopping in \
a cloth bag, a shadow on the ground — not by painting a backdrop behind them. Where the \
original DOES have a setting, that setting becomes Bangladeshi in full.

WHAT MUST NOT CHANGE:
  - THE MESSAGE. Whatever the original teaches the reader, the new picture teaches. The same \
activity, the same situation, the same instruction being illustrated.
  - THE PEOPLE COUNT AND THEIR ROLES. The same number of people, the same ages and genders, \
in the same relationship to each other (a doctor and a patient stay a doctor and a patient). \
Their POSES are deliberately NOT on this list — see the redraw rules above.
  - THE ASPECT RATIO AND SIZE. Return the picture at exactly the aspect ratio and dimensions \
you were given. Do not pad it, do not crop it to a square or a 2:3 frame. It is printed into \
a fixed rectangle on a page.
  - THE DRAWING STYLE. If the original is a flat vector illustration, a line drawing, a \
two-colour graphic or a watercolour, the new picture is the same kind of drawing, at the same \
level of detail. Do not turn an illustration into a photograph or a 3D render, and do not \
turn a photograph into a cartoon.{style}
    It must look like something PRINTED in a booklet, not something generated: no soft glow, \
no bloom, no vignette, no gradient mesh, no drop shadows, no glossy or plastic highlights, no \
depth-of-field blur, no sparkles or bokeh, no over-rendered 3D lighting.
  - THE PALETTE. This picture is printed inside a document and has to look like it belongs to \
the page around it. Use the same hues, the same lightness, the same saturation, the same \
overall tone as the original.{palette} A plain background stays plain and stays its original \
colour; a white background stays white. Do not boost saturation or add new accent colours.
  - THE QUALITY. Cleaner and steadier lines than the original, sharper edges, well-resolved \
detail, no blur and no compression artifacts.

{vocabulary}

CRITICAL — BEHAVIOUR MUST FIT BANGLADESH, NOT ONLY APPEARANCE:
  - MODESTY — everyone is covered. Women wear a saree, a salwar kameez with the orna over the \
chest, or other loose full-length clothing: shoulders, arms, chest, midriff and legs covered. \
Men wear a shirt with full trousers, pyjama or lungi, and are not bare-chested. No shorts, \
vests, sleeveless or low-cut tops, clinging fits, short skirts, swimwear or gym-wear, and no \
bare legs — this holds during exercise, sport, swimming, at the beach and at home. Draw the \
SAME activity in modest, covered clothing rather than dropping the activity.
  - CONTACT BETWEEN MEN AND WOMEN — none. No hugging, kissing, hand-holding, an arm around a \
shoulder or waist, sitting in a lap or leaning on one another. A husband and wife, a carer \
and a patient, a doctor and a patient stand or sit side by side at a respectful distance. \
Contact within the same gender and a parent holding their own young child are normal and stay.
  - MANNERS — eat, give and receive with the right hand; feet stay off tables, chairs and \
desks and soles are not turned towards anyone; people greet with salam or a nod, not a kiss \
or an embrace; older people are shown being deferred to.
  - SETTINGS THAT CARRY BEHAVIOUR — a pub, bar, nightclub, dance floor, sunbathing or beach \
scene becomes the Bangladeshi setting that serves the same purpose: a tea stall, a home \
sitting room, a park or riverside walk, a community hall. No dogs indoors, on a lap or on \
furniture.
  - This never overrides the picture's health message: if the booklet is deliberately showing \
a habit to avoid, it still shows it. Adapt how people behave, not what the page is teaching.

CRITICAL — FOOD RULES. All three apply, in this order:
  1. HALAL ONLY. Never depict pork, ham, bacon, lard, alcohol, beer, wine, or a wine glass. \
If the original shows one, replace it with a halal food filling the same role.
  2. KEEP THE NUTRITIONAL MEANING. A food in a health booklet usually stands for a food group, \
a portion size or a measure. The replacement must be in the SAME food group and show the SAME \
portion: oily fish -> ilish or rui (never dal); wholegrain -> lal chal or atta ruti (never \
white rice); leafy vegetable -> lal shak or palong shak; pulse -> dal; dairy -> doi or milk; \
fruit -> a fruit. Never swap across food groups, and never change how much food is shown or \
how many items are on the plate.
  3. MAKE IT BANGLADESHI. Subject to 1 and 2, everyday Bangladeshi food on Bangladeshi \
crockery, eaten the way it is eaten there.

CRITICAL — NO TEXT:
  - Draw NO text, letters, words, numbers or symbols anywhere, in any script.
  - Every sign, placard, board, label or poster you draw comes back BLANK and CLEAN — empty \
board or paper of a plausible shape, colour and material, with no glyphs of any kind.
  - Signage in a Bangladeshi street scene is drawn as blank painted boards and shutters, not \
as lettering. The booklet's real text is printed separately, so leaving it out is required.

CRITICAL — NO LOGOS:
  - Draw NO logo, wordmark, emblem, crest, badge or institutional mark, and never invent one \
or substitute a real organisation's mark. Leave such areas as clean, empty background.

Focus especially on: {focus}."""

# The cover is a whole printed page, not a picture inside one, and that changes what has to
# be protected. There is no surrounding page for it to match, so the palette instruction is
# about the booklet's identity rather than about blending in; and the title, subtitle and
# publisher lines are real PDF text that will be laid back over this raster, so the areas
# they occupy have to come back as clean, empty background of the original's colour.
COVER_INSTRUCTION = """This is the front cover of a printed health booklet. Redraw it as the \
cover of the same booklet published in Bangladesh, for Bangladeshi readers.
{context_clause}- Keep the EXACT page layout: the same panels, bands and blocks of colour in \
the same places, the same aspect ratio, the same overall design. This must still be \
recognisably the same booklet, not a new design.
- Keep the SAME colour palette as the original — the same background colours, the same \
accent colours, the same tone.{palette} Do not restyle or rebrand it.{style}
- Every person must BE Bangladeshi, not a Western person wearing Bangladeshi clothes. \
Redraw the face itself — South Asian face shape and features, warm brown skin, black or dark \
brown hair, no blond or light hair and no pale complexion — and then dress them (saree, \
salwar kameez, panjabi, kurta, hijab, lungi) as fits their age and role. Keep their poses, \
positions, scale and the original's drawing style.
- Their behaviour must fit Bangladesh as well as their faces. Everyone is modestly covered \
— no shorts, sleeveless or low-cut tops, tight fits, swimwear, gym-wear or bare legs, even \
when exercising — and a man and a woman are not shown touching: no hugging, kissing, \
hand-holding or an arm round a shoulder; they stand side by side at a respectful distance. \
Same-gender contact and a parent holding their own child are normal and stay. Adjust only \
the clothing and the touching; the activity, the count and the layout do not change.
- Any setting, building, vehicle, food or object becomes its Bangladeshi equivalent, drawn \
in the original's style. A pub, bar, dance floor or beach scene becomes the Bangladeshi \
setting that serves the same purpose. Changing the faces and the clothing while the rooms, \
streets, furniture and props stay Western is a failure — the whole cover is localized, \
including what the people are wearing and holding: gloves, hi-vis, coats, scarves, helmets, \
boots and Western bags all go.
{vocabulary}
CRITICAL — NO TEXT ANYWHERE:
  - Draw NO letters, words, numbers, logos or symbols. Not in the title area, not on the \
artwork, not in the footer.
  - Every area that held text must come back as CLEAN, EMPTY background in exactly the \
colour it had — flat and unmarked, ready to be printed over.
  - The booklet's real title is placed back on top of your image afterwards, so leaving \
those areas blank is required, not an omission.
CRITICAL — NO LOGOS:
  - Do NOT redraw, restyle, recolour, translate, or invent any logo, wordmark, emblem, \
crest, badge, or the mark of any hospital, trust, charity, ministry or publisher. Never put \
a different organisation's mark in place of the one that was there.
  - Leave every such area clean and empty in its original background colour. The real marks \
are stamped back over your image afterwards, exactly as they were printed."""

INFORMATION_ROLES = ["decorative", "referential"]

_CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_logo": {"type": "BOOLEAN"},
        "logo_fills_image": {"type": "BOOLEAN"},
        "needs_localization": {"type": "BOOLEAN"},
        "categories": {"type": "ARRAY", "items": {"type": "STRING", "enum": CATEGORIES}},
        "information_role": {"type": "STRING", "enum": INFORMATION_ROLES},
        "reason": {"type": "STRING"},
    },
    "required": [
        "is_logo", "logo_fills_image", "needs_localization", "categories",
        "information_role", "reason",
    ],
}


def classify_image(image_bytes: bytes, mime: str) -> dict:
    """Return {"is_logo": bool, "needs_localization": bool, "categories": [...],
    "information_role": str, "reason": str}.

    On any failure returns needs_localization=False so the image is kept as-is, and
    information_role="referential" so a classifier that answered nothing can never be
    read as permission to redraw the picture.

    `is_logo` is authoritative over everything else: a logo is a real organisation's
    identity, so it is never redrawn and its wording is never translated. The model is asked
    to keep the two answers consistent, but they are reconciled here as well rather than
    trusted — a mark that came back is_logo=true *and* needs_localization=true would
    otherwise be redrawn on the strength of the second answer alone.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[part],
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFY_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_CLASSIFY_SCHEMA,
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "needs_localization" in data:
                data.setdefault("categories", [])
                data.setdefault("reason", "")
                role = str(data.get("information_role") or "").strip().lower()
                data["information_role"] = (
                    role if role in INFORMATION_ROLES else "referential"
                )
                data["is_logo"] = bool(data.get("is_logo"))
                # Missing means "the whole image is the mark", which is the safe reading:
                # a picture wrongly kept whole is un-localized, while one wrongly redrawn
                # misrepresents a real organisation.
                data["logo_fills_image"] = bool(data.get("logo_fills_image", True))
                # Only a picture that IS a mark is taken off the table entirely. One that
                # merely contains a mark is localized normally, with the mark protected by
                # detect_logo_regions — see image_processor._decide_from_png.
                if data["is_logo"] and data["logo_fills_image"]:
                    data["needs_localization"] = False
                    data["categories"] = []
                return data
            logger.warning(
                "Classify: unexpected response shape (attempt %d/%d): %r",
                attempt,
                MAX_ATTEMPTS,
                response.text[:200],
            )
        except Exception:
            logger.exception(
                "Classify request failed (attempt %d/%d)", attempt, MAX_ATTEMPTS
            )
    logger.warning("Classify: giving up — treating image as no-localization")
    return {
        "is_logo": False,
        "logo_fills_image": True,
        "needs_localization": False,
        "categories": [],
        "information_role": "referential",
        "reason": "classify failed",
    }


_TEXT_BLOCKS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "blocks": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "lang": {"type": "STRING"},
                    "box_2d": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["text", "lang", "box_2d"],
            },
        }
    },
    "required": ["blocks"],
}

# Boxes are asked for as `box_2d` — [y0, x0, y1, x1] on a 0-1000 grid — and not as the
# fractions of width and height this function returns, because that is the convention the
# model was trained to answer in and it is markedly better at it.
#
# Measured on the exercise panel from the Revascularisation manual, a 1334x440 graphic
# (3.0:1) holding twelve lines. Asked for fractions it found ten of the twelve and put them
# roughly a line and a half low — "increase blood flow to your heart muscle" came back at
# 0.50-0.58 of the height when it is printed at 0.27-0.33 — and two runs of the identical
# request at temperature 0 disagreed with each other. Asked for box_2d it found all twelve,
# twice, identically, with every line within a few percent of its ink. Nothing else about
# the request changed.
_TEXT_BLOCKS_PROMPT = (
    "Detect every distinct block of visible text in this image (signs, placards, labels, "
    "headings, captions, words). For each block return: 'text' (the exact text as it appears, "
    "one block's lines joined with spaces), 'lang' (ISO code of its language — 'en' for "
    "English, 'bn' for Bangla, 'hi' for Hindi, etc.), and 'box_2d' as [y0, x0, y1, x1] "
    "integers on a 0-1000 grid of the image height/width (y0,x0 = top-left corner, "
    "y1,x1 = bottom-right). Return an empty list if there is no text."
)


# OCR retry budget. Deliberately small and time-boxed: vertex_client already backs off five
# times inside every one of these attempts, and the failure being retried is a 504 on a large
# payload, which more identical requests cannot fix. Three attempts at a 90s ceiling bounds a
# hopeless picture at ~5 minutes instead of the hour that five attempts at the 180s default
# would have cost — on the eatwell plate, measured.
OCR_ATTEMPTS = 3
OCR_TIMEOUT_MS = 90_000
# Longest edge of the copy sent on each successive attempt.
OCR_RETRY_DIMS = (None, 1024, 768)


def _shrunk_for_retry(image_bytes: bytes, mime: str, attempt: int) -> tuple[bytes, str]:
    """The copy to send on `attempt`: the original first, then progressively smaller ones.

    Returns the bytes unchanged if no resize is wanted or possible, so a failure to shrink
    costs an ordinary retry rather than the whole OCR.
    """
    target = OCR_RETRY_DIMS[min(attempt, len(OCR_RETRY_DIMS)) - 1]
    if not target:
        return image_bytes, mime
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if max(img.size) <= target:
                return image_bytes, mime
            img.thumbnail((target, target), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
        logger.info("OCR retry %d: resending at %dpx", attempt, target)
        return buf.getvalue(), "image/png"
    except Exception:
        logger.debug("Could not shrink the image for an OCR retry", exc_info=True)
        return image_bytes, mime


def _extract_text_blocks(image_bytes: bytes, mime: str, notes: dict | None = None) -> list[dict]:
    """Structured OCR: return a list of {"text", "lang", "bbox": [x0,y0,x1,y1]} blocks, with bbox
    normalized to 0..1 of image dimensions.

    Pass `notes` to tell "no text" from "the request failed" apart: `notes` gets
    "ocr_failed": True only when the request failed, and the caller can then decline to
    blank a picture whose words it could not read.
    """
    blocks = _ocr_once(image_bytes, mime)
    if blocks is None:
        if notes is not None:
            notes["ocr_failed"] = True
        return []
    return blocks


def _ocr_once(image_bytes: bytes, mime: str) -> list[dict] | None:
    """One structured-OCR call, retried. None when every attempt failed, [] when there is
    no text — a distinction the callers depend on.

    Both used to be []: a transient error on a picture full of labels read as a picture with
    no labels, and since the edit model is separately told to blank every text surface, the
    words were then deleted rather than translated. That is what happened to the eatwell
    plate's food-group labels: regenerated correctly, captions gone, nothing in the log.
    """
    for attempt in range(1, OCR_ATTEMPTS + 1):
        # Retrying the identical request is the one thing that does not work here: the
        # failure on the biggest pictures is a server-side 504, and the payload is why. Each
        # retry hands over a smaller copy, which is also the cheaper request to serve.
        payload, part_mime = _shrunk_for_retry(image_bytes, mime, attempt)
        try:
            response = generate_content(
                model=CLASSIFY_MODEL,
                contents=[
                    _TEXT_BLOCKS_PROMPT,
                    types.Part.from_bytes(data=payload, mime_type=part_mime),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_TEXT_BLOCKS_SCHEMA,
                    temperature=0.0,
                ),
                timeout_ms=OCR_TIMEOUT_MS,
            )
            data = json.loads(response.text)
            blocks = data.get("blocks", []) if isinstance(data, dict) else []
            cleaned: list[dict] = []
            for b in blocks:
                text = (b.get("text") or "").strip()
                box_2d = b.get("box_2d") or []
                if not text or len(box_2d) != 4:
                    continue
                # box_2d is [y0, x0, y1, x1] on a 0-1000 grid; the rest of the pipeline
                # works in [x0, y0, x1, y1] fractions. Clamped, and empty rects dropped.
                top, left, bottom, right = (
                    min(max(float(v) / 1000.0, 0.0), 1.0) for v in box_2d
                )
                x0, y0, x1, y1 = left, top, right, bottom
                if x1 <= x0 or y1 <= y0:
                    continue
                cleaned.append({
                    "text": text,
                    "lang": (b.get("lang") or "").strip().lower(),
                    "bbox": [x0, y0, x1, y1],
                })
            return cleaned
        except Exception as exc:
            logger.warning(
                "Text-block OCR attempt %d/%d failed: %s",
                attempt, OCR_ATTEMPTS, str(exc)[:200],
            )
    logger.error("Text-block OCR failed after %d attempts — the image's words are unknown",
                 OCR_ATTEMPTS)
    return None


_LOGO_REGIONS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "logos": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["label", "bbox"],
            },
        }
    },
    "required": ["logos"],
}

_LOGO_REGIONS_PROMPT = (
    "Find every logo or brand mark in this image. That means: logos, wordmarks, lettermarks, "
    "emblems, crests, seals, coats of arms, badges and roundels; charity, hospital, trust, "
    "university, government, ministry, NHS, WHO and other institutional marks; an "
    "organisation's name or initials set as its identity, including a plain stylised wordmark; "
    "and any strapline printed as part of such a mark. For each, return 'label' (the "
    "organisation or brand, or a short description if you cannot name it) and 'bbox' as "
    "[x0, y0, x1, y1], each value a fraction between 0 and 1 of the image width/height "
    "(x0,y0 = top-left, x1,y1 = bottom-right). Draw the box tightly around the mark itself, "
    "including its wording, and nothing else. Return an empty list if there are none. Do not "
    "report ordinary headings, captions, body text or page furniture as logos.\n"
    "A mark identifies an ORGANISATION. A slogan, motto or message lettered onto something "
    "inside a picture — words on a character's t-shirt, a hand-written sign, a placard, a "
    "poster, a banner — is not a mark unless an organisation's name or emblem is part of it. "
    "'HELP YOURSELF TO A HEALTHY FUTURE' hand-lettered on a cartoon figure's shirt is a "
    "message to the reader and must NOT be reported; the same shirt carrying 'NHS Lothian' "
    "or a charity's crest must be. Reporting a slogan as a mark takes it out of translation "
    "and deletes it from the page, so when the words name no organisation, leave them out."
)


def detect_logo_regions(image_bytes: bytes, mime: str) -> list[dict]:
    """Locate logos and brand marks in an image: [{"label": str, "bbox": [x0,y0,x1,y1]}].

    bbox values are fractions of the image's width/height, so they map onto any rendering of
    the same picture. Returns [] on no-logos or any failure — a caller that finds no regions
    simply leaves the generated image alone, which is the same outcome as before this
    existed.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    try:
        response = generate_content(
            model=CLASSIFY_MODEL,
            contents=[_LOGO_REGIONS_PROMPT, part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LOGO_REGIONS_SCHEMA,
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        found = data.get("logos", []) if isinstance(data, dict) else []
    except Exception:
        logger.debug("Could not detect logo regions", exc_info=True)
        return []

    cleaned: list[dict] = []
    for entry in found:
        bbox = entry.get("bbox") or []
        if len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        cleaned.append({"label": (entry.get("label") or "").strip(), "bbox": [x0, y0, x1, y1]})
    return cleaned


def _is_valid_bangla(text: str) -> bool:
    """Validate that text contains proper Bangla characters (not hallucinated)."""
    if not text:
        return False
    # Bangla Unicode range: U+0980 to U+09FF
    bangla_chars = sum(1 for c in text if 'ঀ' <= c <= '৿')
    # At least 30% of text should be Bangla characters
    return bangla_chars > 0 and (bangla_chars / len(text)) > 0.3


# The per-block fallback, used when translate_blocks could not answer for a block. It is the
# one translation path with no picture to look at, so the register it has to match — the same
# চলিত/আপনি voice as translator.SYSTEM_PROMPT — has to be stated outright. Left to "natural,
# proper, clear Bangla" it wrote a different, more formal Bangla than the page around it.
_SINGLE_BLOCK_PROMPT = """This is a line of text printed inside a picture in a health booklet \
being republished in Bangladesh. Translate it into Bangla for a patient reading it — an ordinary \
person, often elderly, not a doctor.

- Bangladeshi Bangla, modern colloquial চলিত. Never সাধু ভাষা. Address the reader as আপনি.
- The plainest everyday word, never a bookish one and never a transliterated abbreviation
  (GP is ডাক্তার, never জিপি).
- Say it the way a Bangla speaker would say it, not word by word after the English.
- This is a label or caption, so keep it short — about as long as the English.
- Keep every number, unit and symbol exactly as printed, in Latin digits: "3 units" is
  "3 ইউনিট", never "তিন". A number that has lost its unit is a clinical error in this document.
- A brand, an organisation, a drug name or a person's name stays in Latin script.
- Return only the Bangla. No English, no explanation, no quotation marks.

English: {text}"""


def _translate_to_bangla(text: str) -> str:
    """Translate one English string to Bangla, with retries and a Bangla-output check."""
    if not text or not text.strip():
        return text

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = generate_content(
                model=TEXT_TRANSLATE_MODEL,
                contents=[_SINGLE_BLOCK_PROMPT.format(text=text)],
                # Gemini 3 default. The old ladder climbed 0.1 → 0.3 to shake a
                # different answer out of a retry, which a near-greedy model needs
                # and this one does not: at 1.0 each attempt already differs, and
                # tuning it down degrades the reasoning pass.
                config=types.GenerateContentConfig(temperature=1.0),
            )
            bangla_text = response.text.strip()

            # Validate the translation
            if _is_valid_bangla(bangla_text):
                logger.info("Translated (attempt %d): %s → %s", attempt, text[:50], bangla_text[:50])
                return bangla_text
            else:
                logger.warning(
                    "Translation attempt %d produced invalid Bangla (not enough Bangla chars): %s",
                    attempt, bangla_text[:100]
                )
        except Exception as e:
            logger.warning("Translation attempt %d failed: %s", attempt, str(e)[:200])

    logger.warning("All translation attempts failed for: %s", text[:50])
    return text


def _keeps_numbers(source: str, translated: str) -> bool:
    """True if every run of digits in the source survives into the translation.

    The one check worth making automatically: this booklet keeps its numbers in Latin digits
    (see translator.SYSTEM_PROMPT, "Leave unchanged: numbers, dates"), so a translation that
    dropped one is detectable without knowing any Bangla. "3 units" -> "তিন" fails here, and
    that exact answer shipped on the alcohol-units grid: a card that defines a measure, with
    the measure gone.
    """
    return all(run in translated for run in re.findall(r"\d+(?:[.,]\d+)?", source))


_BLOCK_TRANSLATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "bn": {"type": "STRING"},
                },
                "required": ["index", "bn"],
            },
        }
    },
    "required": ["translations"],
}

_BLOCK_TRANSLATION_PROMPT = """This picture is printed in a health booklet that is being \
republished in Bangladesh. Below is every piece of text printed in it, numbered. Translate \
each one into natural, everyday Bangla for a Bangladeshi patient.

Look at the picture before you answer. It tells you what each piece of text is doing: a \
heading, a caption under a drawing, the name of a food group, a figure printed beside the \
thing it measures.

The reader is a patient — an ordinary person, often elderly, not a doctor — and this text sits \
in a booklet whose pages are already in Bangla. Write the same voice the pages use: Bangladeshi \
modern colloquial চলিত, never সাধু ভাষা, the reader addressed as আপনি, and the plainest everyday \
word rather than a bookish one. Say each label the way a Bangla speaker would say it out loud, \
not word by word after the English. Never use a transliterated English abbreviation — GP is \
ডাক্তার, never জিপি.

Rules:
- Return one entry for every index, with the same index numbers. Never merge two entries, \
never split one, never leave one out.
- KEEP EVERY NUMBER, UNIT, MEASURE AND SYMBOL. "3 units" is "3 ইউনিট", not "তিন". \
"250ml", "12%", "1.5", "80g", "(125ml, ABV 12%)" — the quantity AND its unit both come \
through, written the way they are printed, in the same Latin digits the rest of the booklet \
uses. A number that has lost its unit is a clinical error in this document.
- Translate the whole of a block, including anything inside brackets.
- These blocks all belong to ONE picture. Translate a word that appears in several of them \
the same way every time, and give sibling labels the same style and register — a row of \
tiles has to read as a row.
- An organisation's name, a brand, a drug name or a person's name stays in Latin script.
- Return only the Bangla. No English, no explanation, no quotation marks.
{context_clause}
The text blocks:
{listing}"""


def translate_blocks(
    image_bytes: bytes,
    mime: str,
    blocks: list[dict],
    page_context: str = "",
) -> int:
    """Fill each block's "bn" with its Bangla, in one call that can see the picture.

    Replaces one call per block. The per-block call could not see what it was translating:
    "3 units" came back as "তিন" because nothing in the request said the words were a
    quantity printed beside the drink it measures, and eight tiles of one grid were eight
    separate conversations, so they came back in different registers and most of them not at
    all. One call, with the image and all of the blocks, fixes both — the model sees which
    label is a heading and which is a figure, and it sees its own siblings.

    Returns how many blocks were translated. A block whose answer fails validation is left
    without a "bn", so `resolve_block_text` falls back to the per-block call and then to the
    English: nothing depends on this succeeding.
    """
    pending = [
        (i, (b.get("text") or "").strip())
        for i, b in enumerate(blocks)
        # Nothing to do for a block that is already Bangla, or that a previous call has
        # already answered — the discard path reaches this twice for the same blocks.
        if not _is_valid_bangla((b.get("bn") or "").strip())
    ]
    pending = [(i, t) for i, t in pending if t and not _is_valid_bangla(t)]
    if not pending:
        return 0

    context_clause = ""
    if page_context.strip():
        context_clause = (
            f'\nThe page this picture sits on says: "{page_context.strip()}"\n'
            "Use it only to understand what the words mean. Do not translate it, and do not "
            "add any of it to your answers.\n"
        )

    done = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        listing = "\n".join(f"{i}. {text}" for i, text in pending)
        try:
            response = generate_content(
                model=TEXT_TRANSLATE_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    _BLOCK_TRANSLATION_PROMPT.format(
                        context_clause=context_clause, listing=listing
                    ),
                ],
                config=types.GenerateContentConfig(
                    # Gemini 3 default; the retry no longer needs a temperature
                    # ladder to come back with something different (see
                    # _translate_to_bangla).
                    temperature=1.0,
                    response_mime_type="application/json",
                    response_schema=_BLOCK_TRANSLATION_SCHEMA,
                ),
            )
            answers = json.loads(response.text).get("translations", [])
        except Exception as exc:
            logger.warning(
                "Batch block translation attempt %d failed: %s", attempt, str(exc)[:200]
            )
            continue

        by_index = {int(a.get("index", -1)): (a.get("bn") or "").strip() for a in answers}
        still: list[tuple[int, str]] = []
        for i, text in pending:
            bn = by_index.get(i, "")
            # A block with no letters at all ("12%", "1.5") has nothing to translate into
            # Bangla script, so for those the digits surviving IS the whole test.
            has_letters = any(ch.isalpha() for ch in text)
            ok = bool(bn) and _keeps_numbers(text, bn) and (_is_valid_bangla(bn) or not has_letters)
            if ok:
                blocks[i]["bn"] = bn
                done += 1
            else:
                still.append((i, text))
        if len(still) == len(pending):
            logger.warning(
                "Batch block translation attempt %d validated nothing (%d block(s))",
                attempt, len(pending),
            )
        pending = still

    if pending:
        logger.info(
            "%d of %d text block(s) fell back to per-block translation: %s",
            len(pending), len(pending) + done,
            "; ".join(t[:30] for _, t in pending[:3]),
        )
    return done


def resolve_block_text(block: dict) -> str:
    """Return the text to render for an OCR block: keep it as-is if it is already Bangla,
    otherwise translate it. Returns "" only for an empty block.

    A failed translation falls back to the original words rather than to nothing. By the
    time this is called the baked text has already been painted out of the image, so
    returning "" does not leave the English standing — it deletes it. Untranslated English
    is a shortcoming; a blank where a caption or a dosage figure used to be is a defect.
    """
    text = (block.get("text") or "").strip()
    if not text:
        return ""
    # Already done by the image-aware batch call, which is the better answer because it saw
    # the picture and the block's siblings. image_regen never sets this, so it is unaffected.
    pre = (block.get("bn") or "").strip()
    if pre and _is_valid_bangla(pre):
        return pre
    lang = (block.get("lang") or "").lower()
    if lang.startswith("bn") or _is_valid_bangla(text):
        return text  # already Bangla — keep exactly
    translated = _translate_to_bangla(text)
    if _is_valid_bangla(translated):
        return translated
    logger.warning("Could not translate %r — keeping the original text rather than a blank", text[:60])
    return text


def localize_image(
    image_bytes: bytes,
    mime: str,
    categories: list[str],
    page_context: str = "",
    palette: str = "",
    mode: str = "context",
    notes: dict | None = None,
    style: str = "",
    locks: str = "",
) -> bytes | None:
    """Return edited image bytes adapted to Bangladeshi culture, or None on failure.

    Adapts visual content (people, settings, objects) to Bangladeshi culture.
    Keeps all text from the original image as-is to avoid hallucination.

    `palette` is a measured description of the source's colours (see
    image_processor._palette_summary). Naming the actual hex values holds the model to the
    page's palette far better than asking it to "match the original" — left to itself it
    returns a warmer, more saturated picture that reads as pasted in from another book.

    `style` is the same idea for the drawing technique (image_processor._style_summary): told
    only to "keep the style" the model returns its own — a flat screen-print comes back soft-
    shaded with glow and gradients, and reads as generated.

    `locks` is the reserved-rectangle clause (image_processor._lock_summary): the parts of the
    picture the page's own printed text sits over, which have to come back blank and in place.
    Empty for a picture nothing is printed on.

    `mode` is "context" (the page's own words steer what is drawn), "simple" (a compact
    cultural swap with no page text), or "reimagine" (the scene is drawn again as a
    Bangladeshi one rather than retouched in place). See LOCALIZE_MODES for when each
    applies — "reimagine" needs the frame free, so it is only chosen where nothing has to
    line up with the original afterwards.

    `notes` is a dict the failure reason is written into — pass the caller's audit record so
    that a picture lost to quota is distinguishable from one the model refused.

    None means the caller keeps the original image untouched.
    """
    focus = ", ".join(categories) if categories else "any culturally-specific content"
    palette_note = f"\n  - {palette}" if palette.strip() else ""
    style_note = f"\n    {style}" if style.strip() else ""
    lock_note = f"\n{locks}" if locks.strip() else ""
    context = page_context.strip()

    if mode == "reimagine":
        instruction = REIMAGINE_INSTRUCTION.format(
            focus=focus,
            palette=palette_note,
            style=style_note,
            locks=lock_note,
            vocabulary=BANGLADESHI_VOCABULARY,
            context_clause=(
                REIMAGINE_CONTEXT_CLAUSE.format(context=context) if context else ""
            ),
        )
    elif mode == "simple" or not context:
        instruction = SIMPLE_EDIT_INSTRUCTION.format(
            focus=focus, palette=palette_note, style=style_note, locks=lock_note
        )
    else:
        # The image model must NOT draw any text — it garbles glyphs (especially Bangla) and
        # fights the real text layer. Leave every text surface blank; text is restored
        # afterwards as a Noto overlay (see image_processor._decide / localize_pdf).
        text_instruction = (
            "Leave every sign, placard, label, and text surface completely BLANK — an empty board "
            "or paper of the same shape and color, with no letters, words, numbers, or symbols at "
            "all. Do not translate or re-draw any text; just clear it."
        )
        instruction = EDIT_INSTRUCTION.format(
            focus=focus,
            text_instruction=text_instruction,
            palette=palette_note,
            style=style_note,
            locks=lock_note,
            vocabulary=BANGLADESHI_VOCABULARY,
            context_clause=CONTEXT_CLAUSE.format(context=context),
        )

    return _run_edit_models(instruction, image_bytes, mime, notes)


def localize_cover(
    image_bytes: bytes,
    mime: str,
    page_context: str = "",
    palette: str = "",
    style: str = "",
) -> bytes | None:
    """Return a Bangladeshi-localized rendering of a whole cover page, or None on failure.

    Separate from `localize_image` because a cover is the page, not a picture on it: there
    is no surrounding layout for it to blend into, and the areas its title and publisher
    lines occupy have to come back blank so the real text layer can be printed back over
    them. See COVER_INSTRUCTION.
    """
    context = page_context.strip()
    instruction = COVER_INSTRUCTION.format(
        palette=f"\n  - {palette}" if palette.strip() else "",
        style=f"\n  - {style}" if style.strip() else "",
        vocabulary=BANGLADESHI_VOCABULARY,
        context_clause=CONTEXT_CLAUSE.format(context=context) if context else "",
    )
    return _run_edit_models(instruction, image_bytes, mime)


def _run_edit_models(
    instruction: str,
    image_bytes: bytes,
    mime: str,
    notes: dict | None = None,
    models: list[str] | None = None,
) -> bytes | None:
    """Send one edit instruction to the image model; return image bytes or None.

    `models` defaults to EDIT_MODELS, which is a single model — see the note there. The
    parameter is kept because the loop is written around a list and a second model may be
    reinstated; passing one is how to try an alternative without touching the default.

    `notes` is written into rather than returned — the caller passes its own audit record, so
    why an edit failed lands in the audit with no extra plumbing. Sets "edit_error" to
    "quota", "refused" or "error", and "edit_attempts" to the number of requests made.
    """
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
    contents = [instruction, part]
    chain = models or EDIT_MODELS
    attempts = 0
    outcome = "error"

    # Walk the model fallback chain: try each edit model a couple of times before moving on. This
    # is the recovery path when the primary image model is rate-limited (429) or unavailable —
    # transient backoff is already handled inside vertex_client.generate_content.
    #
    # Then walk the whole chain again, because a picture that failed only because the project
    # was out of quota is not a picture that cannot be localized, and shipping it untouched is
    # exactly the failure this pipeline exists to prevent. By this point vertex_client has
    # already backed off five times and tried both projects, so the only thing left to change
    # is the wait: the image quota is per-MINUTE and shared by every image model, so a pause
    # longer than a minute is the one thing that can actually clear it. Only retried when
    # every failure in the pass was transient — a model that refused will refuse again.
    for extra_pass in range(EDIT_QUOTA_RETRY_PASSES + 1):
        if extra_pass:
            logger.info(
                "Localize: every model was rate-limited; waiting %.0fs for the per-minute "
                "image quota to refill (pass %d of %d)",
                EDIT_QUOTA_COOLDOWN_SEC, extra_pass, EDIT_QUOTA_RETRY_PASSES,
            )
            time.sleep(EDIT_QUOTA_COOLDOWN_SEC)
        all_transient = True
        for model in chain:
            for attempt in range(1, EDIT_ATTEMPTS_PER_MODEL + 1):
                temperature = 0.2 + (attempt - 1) * 0.3  # 0.2, 0.5, ...
                attempts += 1
                try:
                    logger.info("Image generation: model=%s attempt %d/%d (temp=%.1f)",
                                model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature)
                    response = generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            temperature=temperature,
                        ),
                        # Image generation is legitimately slower than text; give it a wider
                        # ceiling than the client-level default so a real render isn't aborted.
                        timeout_ms=IMAGE_TIMEOUT_MS,
                    )
                    out = _first_image_bytes(response)
                    if out:
                        logger.info(
                            "Image generation succeeded with %s on attempt %d", model, attempt
                        )
                        if notes is not None:
                            notes["edit_attempts"] = attempts
                        return out
                    logger.warning(
                        "Localize: no image from %s (attempt %d/%d, temp=%.1f)",
                        model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature,
                    )
                    _log_response_diagnostics(response, attempt)
                    # A response that came back without an image is a refusal, not a queue.
                    all_transient = False
                    outcome = "refused"
                except Exception as e:
                    logger.warning(
                        "Localize request failed on %s (attempt %d/%d, temp=%.1f): %s",
                        model, attempt, EDIT_ATTEMPTS_PER_MODEL, temperature, str(e)[:200],
                    )
                    if _is_quota_error(e):
                        if outcome != "refused":
                            outcome = "quota"
                    else:
                        all_transient = False
                        outcome = "refused" if outcome == "refused" else "error"
            logger.info("Localize: model %s exhausted; trying next fallback model", model)
        if not all_transient:
            break  # a refusal will not become an acceptance by waiting

    logger.warning(
        "Localize: all edit models failed after %d request(s) (%s) — keeping original image",
        attempts, outcome,
    )
    if notes is not None:
        notes["edit_error"] = outcome
        notes["edit_attempts"] = attempts
    return None


# Markers of a "come back later" failure, as opposed to a refusal. Mirrors
# vertex_client._TRANSIENT_MARKERS, kept short here because only the quota case earns a wait.
_QUOTA_MARKERS = ("resource_exhausted", "429", "quota", "rate limit", "unavailable", "503")


def _is_quota_error(exc: Exception) -> bool:
    """True if an edit failure is a rate limit rather than a refusal."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _first_image_bytes(response) -> bytes | None:
    """Pull the first inline image payload out of a generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def _log_response_diagnostics(response, attempt: int) -> None:
    """Log diagnostics when a response doesn't contain an image."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        logger.debug("Attempt %d: no candidates in response", attempt)
        return

    for i, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason and finish_reason != "STOP":
            logger.warning("Attempt %d: candidate %d finish_reason=%s", attempt, i, finish_reason)

    feedback = getattr(response, "prompt_feedback", None)
    if feedback:
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            logger.warning("Attempt %d: prompt_feedback.block_reason=%s", attempt, block_reason)
        safety_ratings = getattr(feedback, "safety_ratings", None)
        if safety_ratings:
            for rating in safety_ratings:
                category = getattr(rating, "category", None)
                probability = getattr(rating, "probability", None)
                blocked = getattr(rating, "blocked", None)
                if blocked or probability == "HIGH":
                    logger.warning(
                        "Attempt %d: safety rating %s=%s, blocked=%s",
                        attempt,
                        category,
                        probability,
                        blocked,
                    )
