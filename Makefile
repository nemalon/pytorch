
all: alloc

alloc: alloc.cc
	g++ alloc.cc -o alloc.so -I/usr/lib/cuda/include  -shared -fPIC -lcuda 


test: alloc
	python test_allocator.py
