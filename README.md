## Párhuzamos Eszközök Programozása Project
Benedek Gábor (ABCHLD)

## Leírás

A program egy felhasználó által kiválasztott módon megadott szöveget alakít át a Huffman kódolás szerint.<br>
A program szekvenciális és OpenCL megoldással is szolgál az alábbi részekhez:<br>
- Megadott byte hosszúságú irányítottan véletlenszerű karaktersor generálása
- A karakterek gyakoriságának kiszámolása<br>

(A program méri az egyes részek futási idejét, hogy össze lehessen hasonlítani azokat)

## Követelmények

- GCC (v9+)
- OpenCL fejlesztői könyvtár és driver
- Make

## Mappastruktúra

```

├── include/                   # Fejlécek (kernel\_loader.h, huffman.h)
├── src/
│   ├── main.c                 # Főprogram
│   ├── kernel\_loader.c       # OpenCL kernel betöltése
│   └── huffman.c              # Huffman-algoritmus
├── kernels/
│   ├── byte\_frequency.cl
│   └── random\_generator.cl
├──  measurement/             # Mérések eredményei
│   └── osszehasonlitas.xlsx  # Szekvenciális és OpenCL futási idők összehasonlítása
├── input/                    # Bemeneti állományok (input.txt)
├── output/                   # Kimeneti fájlok (output.txt, eredmények)
├── build/                    # Fordított állományok (main.exe)
└── Makefile                  # Dokumentáció

````

## Fordítás és futtatás

```bash
# Teljes tisztítás és fordítás
make all

# A build/main.exe rögtön el is indul.
````

## Használat

A program indításkor kéri a módot:

* `manual`:

  1. Bemeneti forrás kiválasztása 
  2. Fájlba írási opciók
* `test`:
  – Exponenciális méretsorozaton méri a generálásának és a byte-gyakoriság kiszámolásának idejét, valamint a tömörítés hatékonyságát.
  – Eredmények `.txt` fájlokba íródnak a `output/` mappában.

```text
Select mode [manual/test]:
```

### Manual mód

1. Bemenetet választása:
   - `input/input.txt`
   - A program által enerált karaktersor
   - A Console ablakban megadott szöveg
3. Megjeleníti a leggyakoribb byte-okat és kódjaikat, valamint a tömörítés hatékonyságát
4. Kimenet: `output/output.txt` vagy képernyő

### Test mód

– Automatikus benchmark a `output/` mappába:

* `generation_results.txt`
* `byte_frequencies_results.txt`
* `compression_results.txt`

## Tisztítás

```bash
make clean
```

Eltávolítja a `build/` mappa tartalmát és üríti a konzolt.
