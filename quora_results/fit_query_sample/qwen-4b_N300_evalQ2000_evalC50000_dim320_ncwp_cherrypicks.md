# NCWP Cherry-picked Examples (quora, qwen-4b)

- Fit sample N: `300`
- Eval: `evalQ=2000`, `evalC=50000`
- Compared dimension: `320`
- Selection mode: `strict`
- Criteria:
  - strict: Base / PCA-White / Soft-White first relevant rank > 10, NCWP first relevant rank <= 3
  - relaxed fallback: Base / PCA-White / Soft-White first relevant rank > 10, NCWP first relevant rank <= 10

## Example 1

- Query ID: `306365`
- Query: Why dopeople can't sleep with lights on?
- First relevant rank: Base `941`, PCA `1623`, Soft `252`, NCWP `2`
- NCWP best relevant doc: `306364` - Why can't I sleep with the lights off?

### Base Top-5
1. `403001` - What doesit take for a man to propose?
2. `486969` - Does it mean that you are tired or ill if you can't get your energy back in minutes?
3. `14479` - Why don't you cry?
4. `1638` - What does "I fancy you" mean?
5. `243298` - What do you mean by "hi"?

### PCA-White Top-5
1. `353059` - Bendix square employeeservicepinwithsmallrubylookingstonewhatyearisit?
2. `403001` - What doesit take for a man to propose?
3. `291121` - What is the difference between "trip", "tour" and "journey"?
4. `337083` - What does "pun" mean?
5. `449384` - Is the word "overrated" overrated?

### Soft-White Top-5
1. `353059` - Bendix square employeeservicepinwithsmallrubylookingstonewhatyearisit?
2. `403001` - What doesit take for a man to propose?
3. `14479` - Why don't you cry?
4. `174268` - How can one "let loose"?
5. `337083` - What does "pun" mean?

### NCWP Top-5
1. `403001` - What doesit take for a man to propose?
2. `306364` [relevant] - Why can't I sleep with the lights off?
3. `498250` - Why do people sleep with each other?
4. `353059` - Bendix square employeeservicepinwithsmallrubylookingstonewhatyearisit?
5. `14479` - Why don't you cry?

## Example 2

- Query ID: `135674`
- Query: What is the easiest way to learn PPC?
- First relevant rank: Base `288`, PCA `99`, Soft `128`, NCWP `2`
- NCWP best relevant doc: `135673` - How can i learn PPC?

### Base Top-5
1. `260728` - What is the easiest way to learn good photography?
2. `44200` - What is the the best way to learn programming?
3. `113315` - What is the best way to start learn hacking?
4. `515642` - What is the best way to learn SAS?
5. `355293` - What is the best way to learn analytics?

### PCA-White Top-5
1. `260728` - What is the easiest way to learn good photography?
2. `194366` - What is the fastest and easiest way to learn piano?
3. `156451` - What is the easiest way to suicide?
4. `339765` - What is the easiest way to get in shape?
5. `131567` - What is the easiest way to learn about laws in India?

### Soft-White Top-5
1. `260728` - What is the easiest way to learn good photography?
2. `194366` - What is the fastest and easiest way to learn piano?
3. `44200` - What is the the best way to learn programming?
4. `5520` - What is the easiest way to improve my vocabulary?
5. `156451` - What is the easiest way to suicide?

### NCWP Top-5
1. `260728` - What is the easiest way to learn good photography?
2. `135673` [relevant] - How can i learn PPC?
3. `328437` - What is the easiest way to become wealthy?
4. `181116` - What is the best way to learn electrical engineering?
5. `339765` - What is the easiest way to get in shape?

## Example 3

- Query ID: `52494`
- Query: What's the big deal about cultural appropriation?
- First relevant rank: Base `1034`, PCA `153`, Soft `67`, NCWP `1`
- NCWP best relevant doc: `52495` - Why do people care so much about cultural appropriation?

### Base Top-5
1. `467943` - What's the difference between culture and tradition?
2. `192963` - What's the point of life?
3. `450846` - What's the relationship between science and belief?
4. `205473` - What is the big deal about working at Google?
5. `170401` - What's special about the Arabic language?

### PCA-White Top-5
1. `467943` - What's the difference between culture and tradition?
2. `192963` - What's the point of life?
3. `430332` - What's the difference between kidnapping and abduction?
4. `446295` - What's the effect of coffee?
5. `386950` - What's the difference between stupidity and learning disability?

### Soft-White Top-5
1. `467943` - What's the difference between culture and tradition?
2. `192963` - What's the point of life?
3. `395845` - What's the meaning of this?
4. `450846` - What's the relationship between science and belief?
5. `386950` - What's the difference between stupidity and learning disability?

### NCWP Top-5
1. `52495` [relevant] - Why do people care so much about cultural appropriation?
2. `467943` - What's the difference between culture and tradition?
3. `40232` [relevant] - What is cultural appropriation and why is it such a big problem?
4. `336696` - What is the definition of cultural boundary?
5. `17587` - What are cultural faux pas?

## Example 4

- Query ID: `254177`
- Query: What's the best way to treat roach bites?
- First relevant rank: Base `1842`, PCA `145`, Soft `67`, NCWP `2`
- NCWP best relevant doc: `254178` - How do I treat a sore roach bite?

### Base Top-5
1. `465558` - What is the best way to kill house flies?
2. `33601` - What is the best way to cure of anxiety disorder?
3. `7166` - What is the best way to get rid of acne?
4. `183857` - What is the best way to get rid of a scar?
5. `207980` - What is the best way to get rid of annoying roommates?

### PCA-White Top-5
1. `313775` - What's the best way to learn chess?
2. `537212` - What's the best way to deal intimidating people?
3. `465558` - What is the best way to kill house flies?
4. `7166` - What is the best way to get rid of acne?
5. `79768` - What's the best way to mince garlic?

### Soft-White Top-5
1. `258914` - What is the best way to treat the flu?
2. `287547` - What's the best way to deal with violent people?
3. `79768` - What's the best way to mince garlic?
4. `465558` - What is the best way to kill house flies?
5. `537212` - What's the best way to deal intimidating people?

### NCWP Top-5
1. `487101` - How do you get rid of roaches?
2. `254178` [relevant] - How do I treat a sore roach bite?
3. `531660` - What's the best way to get sleep?
4. `258914` - What is the best way to treat the flu?
5. `33601` - What is the best way to cure of anxiety disorder?

## Example 5

- Query ID: `41356`
- Query: Will Ukraine ever join Nato?
- First relevant rank: Base `98`, PCA `131`, Soft `55`, NCWP `2`
- NCWP best relevant doc: `41355` - What are the chances of Ukraine joining NATO?

### Base Top-5
1. `103356` - Will Russia invade Britain?
2. `301297` - Will Russia destroy Isis?
3. `215069` - Why doesn’t Ukraine join NATO?
4. `271124` - Why Ukraine isn't joining Russia?
5. `237724` - Will India and Pakistan go to war again?

### PCA-White Top-5
1. `103356` - Will Russia invade Britain?
2. `301297` - Will Russia destroy Isis?
3. `464705` - Will Pakistan nuke India?
4. `233455` - Will India Balkanize?
5. `458786` - Why did Russia annex Crimea?

### Soft-White Top-5
1. `103356` - Will Russia invade Britain?
2. `301297` - Will Russia destroy Isis?
3. `233455` - Will India Balkanize?
4. `385521` - Will Scotland and Northern Ireland leave the UK now?
5. `464705` - Will Pakistan nuke India?

### NCWP Top-5
1. `215069` - Why doesn’t Ukraine join NATO?
2. `41355` [relevant] - What are the chances of Ukraine joining NATO?
3. `301297` - Will Russia destroy Isis?
4. `271124` - Why Ukraine isn't joining Russia?
5. `103356` - Will Russia invade Britain?

## Example 6

- Query ID: `327509`
- Query: Which is the best online news portal?
- First relevant rank: Base `1266`, PCA `52`, Soft `200`, NCWP `3`
- NCWP best relevant doc: `327508` - What are the best online news portal?

### Base Top-5
1. `246898` - Which is the best website for writing blogs?
2. `226086` - Which is the best website for finding jobs?
3. `275982` - Which is the best website development software?
4. `112256` - Which is the best news channel to watch?
5. `457593` - Which is the best video calling app?

### PCA-White Top-5
1. `112256` - Which is the best news channel to watch?
2. `93720` - Which is the best online shopping website in Singapore?
3. `226086` - Which is the best website for finding jobs?
4. `246898` - Which is the best website for writing blogs?
5. `275982` - Which is the best website development software?

### Soft-White Top-5
1. `226086` - Which is the best website for finding jobs?
2. `112256` - Which is the best news channel to watch?
3. `246898` - Which is the best website for writing blogs?
4. `93720` - Which is the best online shopping website in Singapore?
5. `275982` - Which is the best website development software?

### NCWP Top-5
1. `112256` - Which is the best news channel to watch?
2. `93720` - Which is the best online shopping website in Singapore?
3. `327508` [relevant] - What are the best online news portal?
4. `108995` - Which is the best Android games?
5. `359371` - Which is the best flight simulator?

## Example 7

- Query ID: `153281`
- Query: Is the Ancient Alien Theory Correct?
- First relevant rank: Base `1092`, PCA `219`, Soft `49`, NCWP `3`
- NCWP best relevant doc: `338886` - What are the evidences to support the ancient alien theory?

### Base Top-5
1. `196126` - Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise?
2. `105616` - Is the New York Times failing?
3. `244016` - Is the Bible a lie?
4. `321281` - Is the Moon hollow?
5. `395094` - Does anyone take the The Church of the SubGenius, a satire, seriously?

### PCA-White Top-5
1. `430831` - Is Educational Psychology a Science of Education?
2. `196126` - Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise?
3. `321281` - Is the Moon hollow?
4. `311632` - Is The Secret real? Does it work?
5. `244016` - Is the Bible a lie?

### Soft-White Top-5
1. `196126` - Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise?
2. `244016` - Is the Bible a lie?
3. `430831` - Is Educational Psychology a Science of Education?
4. `321281` - Is the Moon hollow?
5. `532863` - Is the Bermuda Triangle actually dangerous?

### NCWP Top-5
1. `244016` - Is the Bible a lie?
2. `196126` - Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise?
3. `338886` [relevant] - What are the evidences to support the ancient alien theory?
4. `357704` - How historically accurate is the evidence shown in "Ancient Aliens" from History Channel?
5. `199715` - Is the Big Bang actually possible, or not?

## Example 8

- Query ID: `312568`
- Query: What are the benefits and detriments of neutering?
- First relevant rank: Base `78`, PCA `98`, Soft `49`, NCWP `3`
- NCWP best relevant doc: `155514` - What is the benefit of dog neutering?

### Base Top-5
1. `22954` - What are the pros and cons of marriage?
2. `488931` - What are the advantages and disadvantages of marriage?
3. `411464` - What are the advantages and disadvantages of cannabis?
4. `204975` - What are the pros and cons of friction?
5. `177292` - What are the pros and cons of nationalism?

### PCA-White Top-5
1. `22954` - What are the pros and cons of marriage?
2. `162097` - What are the pros and cons of elaborate weaving?
3. `247014` - What are the pros and cons of using steroids?
4. `222628` - What are the pros and cons of transpirational pulls?
5. `305396` - What are the benefits of celibacy?

### Soft-White Top-5
1. `22954` - What are the pros and cons of marriage?
2. `411464` - What are the advantages and disadvantages of cannabis?
3. `488931` - What are the advantages and disadvantages of marriage?
4. `268715` - What are the advantages/disadvantages of eating potatoes?
5. `335709` - What are the advantages and disadvantages of Democracy?

### NCWP Top-5
1. `365883` - What are the benefits of bidding?
2. `18737` - What are the benefits and tradeoffs of pair programming?
3. `155514` [relevant] - What is the benefit of dog neutering?
4. `488931` - What are the advantages and disadvantages of marriage?
5. `229086` - What are the advantages and disadvantages of using oil?
