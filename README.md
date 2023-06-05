# Libxc tutorial
Directory for Libxc totorial

We have following example systems

    1. Graphene: metalic system with occupation smearing
    2. WS2     : insulating system w/o occupation smearing

For each example system, we have LDA/GGA example directries.

In each directory, we compute

    1. Vxc(r) and  <mk|Vxc|nk> from pw2bgw.x ... Ref
    2. Vxc(r) and  <mk|Vxc|nk> from pp.x
    3. Vxc(r) and  <mk|Vxc|nk> from Libxc

We only consider collinear calculation.


For method 2.

    we first do pp.x and print Vxc(r) = Vtot(r) - Vbare(r) - Vhatree(r) in a FFTgrid and then
        
    1-1. do FFT Vxc(r) to Vxc(g). Then we can compute <mk|Vxc|nk> in G-space.
    1-2. do IFFT unk(g) to unk(r). Then we can compute <mk|Vxc|nk> in R-space.
