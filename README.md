# G-Tra-POPTICS
Attempt at a faithful implementation of the G-Tra-POPTICS GPU trajectory clustering algorithm from "A scalable and fast OPTICS for clustering trajectory big data" by Deng et. al. from 2015.

After working on this implementation for several weeks, I determined that the G-Tra-POPTICS paper does not include enough information to implement their algorithm. Specifically the psuedocode proposed to generate minimum spanning trees in parallel simply won't be correct, based on the psudocode provided. Additionally the FindNeighbors psuedocode provided will not correctly consider neighbors in other data partitions. 
