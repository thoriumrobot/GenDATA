/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public final class SynchronizedSymbolTable extends SymbolTable {

    @Positive
    protected SymbolTable fSymbolTable;

    @Positive
    public SynchronizedSymbolTable(SymbolTable symbolTable) {
    @Positive
    }

    @Positive
    public SynchronizedSymbolTable() {
    @Positive
    }

    @Positive
    public SynchronizedSymbolTable(int size) {
    @Positive
    }

    @Positive
    public String addSymbol(String symbol);

    @Positive
    public String addSymbol(char[] buffer, int offset, int length);

    @Positive
    @Pure
    @Positive
    public boolean containsSymbol(String symbol);

    @Positive
    @Pure
    @Positive
    public boolean containsSymbol(char[] buffer, int offset, int length);
    @Positive
}

// CFWR semantic augmentation - variant 0
