CC=gcc
EXE=tloop segfault forever

all: $(EXE)

tloop: tloop.c
	$(CC) $< -o $@

forever: forever.c
	$(CC) $< -o $@

segfault: segfault.c
	$(CC) $< -o $@

clean:
	rm $(EXE)
