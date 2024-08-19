default: make

alias c := clean
alias f := flamegraph
alias i := interactive
alias m := make
alias p := plop
alias pv := plop_view
alias s := shell
alias t := test

test_dir := justfile_directory() + "/tests/"
ffi_dir := justfile_directory() + "/komrs"

clean:
	-rm "{{ ffi_dir }}"/*.so

make:
	poetry run maturin develop

flamegraph arg:
	poetry run python3 -m plop.collector -f flamegraph {{ arg }}

install:
	poetry install
	cargo install maturin
	@just make

plop arg:
	poetry run python3 -m plop.collector {{ arg }}

plop_view:
	poetry run python3 -m plop.viewer --datadir="{{ justfile_directory() + '/kom_py/profiles' }}"

[no-cd]
shell *kargs:
	poetry run python3 {{ kargs }}

[no-cd]
interactive *kargs:
	poetry run python3 -i {{ kargs }}

test:
    @just make
    poetry run pytest "{{ test_dir }}"
