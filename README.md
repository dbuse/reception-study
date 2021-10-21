Reception Study
===============

A quick study on how reception is influenced by distance and interference, and how Veins parameters like `maxInterferenceDistance` play into this.


Goals
-----

- get a better understanding of:
	- how far signals reach
	- how strong they have to be to be decoded at the receiver
	- how MCS and other parameters influence this
	- and **especially** how much influence interference has
- to find the "right" values for `maxInterferenceDistance` and see what happens if it is too low


Progress so far
---------------

I started with a simple notebook and the idea to re-implement simple pathloss and the NIST packet error model in it.
The idea was to play around with some paraterers and get a better understanding of the whole thing.
This then evolved as the implementation had some flaws and did not match what other sources said, especially for the NIST model.

I thus re-read [Basti's 2019 "A Case for Good Defaults: Pitfalls in VANET Physical Layer Simulations" paper](https://www.bastibl.net/bib/bloessl2019case/).
It showed was a nice introduction to how the whole process works (this should all still be valid for current versions of Veins, except for some parameters).


Next Steps and furhter Ideas
----------------------------

- plan a little simulation experiment (like Basti in 2019) to practically check how transmissions behave
	- a stationary car (sending messages) and one driving away from it (receiving those messages)
	- evaluating received signal strength, specific SNR, packet errors
	- with increasing distances between the vehicles (through slow movement or so)
	- varying parameters like transmit power, noise (to also simulate interference, as they are computationally the same)
	- consider messages also beyond where they can be decoded, to record potential interference
- maybe: fix the implementation in the notebook to play with results ad-hoc
- sketch out how much interference could be possible without detection
	- without obstacles, so with a "working network"
	- thus only with hidden terminal problems through distance
	- kind of construct a "worst case valid scenario"
	- answering the question: how many senders can create interference at the same time, without CSMA/CA stopping them
	- this should give me some insights on how strong interference could be, and how distance plays into this
	- also with varying parameters for transmit power and so on
	- later maybe plan an extension with some kind of obstacles (road canyons?)

Literature
----------

- [Bastian Bloessl and Aisling O'Driscoll, "A Case for Good Defaults: Pitfalls in VANET Physical Layer Simulations," Proceedings of IFIP Wireless Days (WD 2019), Manchester, United Kingdom, April 2019. ](https://www2.tkn.tu-berlin.de/bib/bloessl2019case/)
- [Thales Teixeira de Almeida, Lucas de Carvalho Gomes, Fernando Molano Ortiz, José Geraldo Ribeiro Júnior and Luis Henrique M. K. Consta, "Comparative Analysis of a Vehicular Safety Application in NS-3 and Veins," IEEE Transactions on Intelligent Transportation Systems (TITS), August 2020. ](https://www2.tkn.tu-berlin.de/bib/teixeirade2020comparative/)
