package com.example;

import java.util.List;

// Suppose this is an isolated Java program from a real life project, and all of the implementation
// for methods from List are in Baz. Because Baz is present in the input, Specimin can figure out
// that the call to add() refers to List.add (using the implementation in Baz).
class Foo<E> extends Baz<E> implements List<E> {
    public void bar(E input) {
        this.add(input);
    }

}
