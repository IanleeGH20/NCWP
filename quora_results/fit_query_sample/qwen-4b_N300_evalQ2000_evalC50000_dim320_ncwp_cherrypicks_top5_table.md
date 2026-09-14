# NCWP Cherry-picked Top-5 Table (quora, qwen-4b)

- Fit sample N: `300`
- Eval: `evalQ=2000`, `evalC=50000`
- Compared dimension: `320`
- Selection mode: `strict`

## Example 1

- Query ID: `306365`
- Query: Why dopeople can't sleep with lights on?
- First relevant rank: Base `941`, PCA `1623`, NCWP `2`


| Method    | Top-1                                                                           | Top-2                                                                                         | Top-3                                                                 | Top-4                                                                           | Top-5                                       |
| --------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| Base      | `403001` What doesit take for a man to propose?                                 | `486969` Does it mean that you are tired or ill if you can't get your energy back in minutes? | `14479` Why don't you cry?                                            | `1638` What does "I fancy you" mean?                                            | `243298` What do you mean by "hi"?          |
| PCA-White | `353059` Bendix square employeeservicepinwithsmallrubylookingstonewhatyearisit? | `403001` What doesit take for a man to propose?                                               | `291121` What is the difference between "trip", "tour" and "journey"? | `337083` What does "pun" mean?                                                  | `449384` Is the word "overrated" overrated? |
| NCWP      | `403001` What doesit take for a man to propose?                                 | `306364` [relevant] Why can't I sleep with the lights off?                                    | `498250` Why do people sleep with each other?                         | `353059` Bendix square employeeservicepinwithsmallrubylookingstonewhatyearisit? | `14479` Why don't you cry?                  |


## Example 2

- Query ID: `135674`
- Query: What is the easiest way to learn PPC?
- First relevant rank: Base `288`, PCA `99`, NCWP `2`


| Method    | Top-1                                                       | Top-2                                                        | Top-3                                                 | Top-4                                                          | Top-5                                                          |
| --------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Base      | `260728` What is the easiest way to learn good photography? | `44200` What is the the best way to learn programming?       | `113315` What is the best way to start learn hacking? | `515642` What is the best way to learn SAS?                    | `355293` What is the best way to learn analytics?              |
| PCA-White | `260728` What is the easiest way to learn good photography? | `194366` What is the fastest and easiest way to learn piano? | `156451` What is the easiest way to suicide?          | `339765` What is the easiest way to get in shape?              | `131567` What is the easiest way to learn about laws in India? |
| NCWP      | `260728` What is the easiest way to learn good photography? | `135673` [relevant] How can i learn PPC?                     | `328437` What is the easiest way to become wealthy?   | `181116` What is the best way to learn electrical engineering? | `339765` What is the easiest way to get in shape?              |


## Example 3

- Query ID: `52494`
- Query: What's the big deal about cultural appropriation?
- First relevant rank: Base `1034`, PCA `153`, NCWP `1`


| Method    | Top-1                                                                       | Top-2                                                         | Top-3                                                                               | Top-4                                                  | Top-5                                                                     |
| --------- | --------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| Base      | `467943` What's the difference between culture and tradition?               | `192963` What's the point of life?                            | `450846` What's the relationship between science and belief?                        | `205473` What is the big deal about working at Google? | `170401` What's special about the Arabic language?                        |
| PCA-White | `467943` What's the difference between culture and tradition?               | `192963` What's the point of life?                            | `430332` What's the difference between kidnapping and abduction?                    | `446295` What's the effect of coffee?                  | `386950` What's the difference between stupidity and learning disability? |
| NCWP      | `52495` [relevant] Why do people care so much about cultural appropriation? | `467943` What's the difference between culture and tradition? | `40232` [relevant] What is cultural appropriation and why is it such a big problem? | `336696` What is the definition of cultural boundary?  | `17587` What are cultural faux pas?                                       |


## Example 4

- Query ID: `254177`
- Query: What's the best way to treat roach bites?
- First relevant rank: Base `1842`, PCA `145`, NCWP `2`


| Method    | Top-1                                              | Top-2                                                     | Top-3                                              | Top-4                                               | Top-5                                                           |
| --------- | -------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Base      | `465558` What is the best way to kill house flies? | `33601` What is the best way to cure of anxiety disorder? | `7166` What is the best way to get rid of acne?    | `183857` What is the best way to get rid of a scar? | `207980` What is the best way to get rid of annoying roommates? |
| PCA-White | `313775` What's the best way to learn chess?       | `537212` What's the best way to deal intimidating people? | `465558` What is the best way to kill house flies? | `7166` What is the best way to get rid of acne?     | `79768` What's the best way to mince garlic?                    |
| NCWP      | `487101` How do you get rid of roaches?            | `254178` [relevant] How do I treat a sore roach bite?     | `531660` What's the best way to get sleep?         | `258914` What is the best way to treat the flu?     | `33601` What is the best way to cure of anxiety disorder?       |


## Example 5

- Query ID: `41356`
- Query: Will Ukraine ever join Nato?
- First relevant rank: Base `98`, PCA `131`, NCWP `2`


| Method    | Top-1                                   | Top-2                                                            | Top-3                                   | Top-4                                      | Top-5                                             |
| --------- | --------------------------------------- | ---------------------------------------------------------------- | --------------------------------------- | ------------------------------------------ | ------------------------------------------------- |
| Base      | `103356` Will Russia invade Britain?    | `301297` Will Russia destroy Isis?                               | `215069` Why doesn’t Ukraine join NATO? | `271124` Why Ukraine isn't joining Russia? | `237724` Will India and Pakistan go to war again? |
| PCA-White | `103356` Will Russia invade Britain?    | `301297` Will Russia destroy Isis?                               | `464705` Will Pakistan nuke India?      | `233455` Will India Balkanize?             | `458786` Why did Russia annex Crimea?             |
| NCWP      | `215069` Why doesn’t Ukraine join NATO? | `41355` [relevant] What are the chances of Ukraine joining NATO? | `301297` Will Russia destroy Isis?      | `271124` Why Ukraine isn't joining Russia? | `103356` Will Russia invade Britain?              |


## Example 6

- Query ID: `327509`
- Query: Which is the best online news portal?
- First relevant rank: Base `1266`, PCA `52`, NCWP `3`


| Method    | Top-1                                                 | Top-2                                                           | Top-3                                                     | Top-4                                                 | Top-5                                                    |
| --------- | ----------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| Base      | `246898` Which is the best website for writing blogs? | `226086` Which is the best website for finding jobs?            | `275982` Which is the best website development software?  | `112256` Which is the best news channel to watch?     | `457593` Which is the best video calling app?            |
| PCA-White | `112256` Which is the best news channel to watch?     | `93720` Which is the best online shopping website in Singapore? | `226086` Which is the best website for finding jobs?      | `246898` Which is the best website for writing blogs? | `275982` Which is the best website development software? |
| NCWP      | `112256` Which is the best news channel to watch?     | `93720` Which is the best online shopping website in Singapore? | `327508` [relevant] What are the best online news portal? | `108995` Which is the best Android games?             | `359371` Which is the best flight simulator?             |


## Example 7

- Query ID: `153281`
- Query: Is the Ancient Alien Theory Correct?
- First relevant rank: Base `1092`, PCA `219`, NCWP `3`


| Method    | Top-1                                                                                  | Top-2                                                                                  | Top-3                                                                           | Top-4                                                                                              | Top-5                                                                           |
| --------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Base      | `196126` Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise? | `105616` Is the New York Times failing?                                                | `244016` Is the Bible a lie?                                                    | `321281` Is the Moon hollow?                                                                       | `395094` Does anyone take the The Church of the SubGenius, a satire, seriously? |
| PCA-White | `430831` Is Educational Psychology a Science of Education?                             | `196126` Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise? | `321281` Is the Moon hollow?                                                    | `311632` Is The Secret real? Does it work?                                                         | `244016` Is the Bible a lie?                                                    |
| NCWP      | `244016` Is the Bible a lie?                                                           | `196126` Is Scott Gordon's "The God Entity: The Theory of Everything" a valid premise? | `338886` [relevant] What are the evidences to support the ancient alien theory? | `357704` How historically accurate is the evidence shown in "Ancient Aliens" from History Channel? | `199715` Is the Big Bang actually possible, or not?                             |


## Example 8

- Query ID: `312568`
- Query: What are the benefits and detriments of neutering?
- First relevant rank: Base `78`, PCA `98`, NCWP `3`


| Method    | Top-1                                           | Top-2                                                            | Top-3                                                           | Top-4                                                           | Top-5                                                            |
| --------- | ----------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Base      | `22954` What are the pros and cons of marriage? | `488931` What are the advantages and disadvantages of marriage?  | `411464` What are the advantages and disadvantages of cannabis? | `204975` What are the pros and cons of friction?                | `177292` What are the pros and cons of nationalism?              |
| PCA-White | `22954` What are the pros and cons of marriage? | `162097` What are the pros and cons of elaborate weaving?        | `247014` What are the pros and cons of using steroids?          | `222628` What are the pros and cons of transpirational pulls?   | `305396` What are the benefits of celibacy?                      |
| NCWP      | `365883` What are the benefits of bidding?      | `18737` What are the benefits and tradeoffs of pair programming? | `155514` [relevant] What is the benefit of dog neutering?       | `488931` What are the advantages and disadvantages of marriage? | `229086` What are the advantages and disadvantages of using oil? |


