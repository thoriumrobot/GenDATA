/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.utils;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;

@AnnotatedFor({ "signedness" })
public class UnsyncBufferedOutputStream extends FilterOutputStream {

    protected byte[] buffer;

    protected int count;

    public UnsyncBufferedOutputStream(OutputStream out) {
    }

    public UnsyncBufferedOutputStream(OutputStream out, int size) {
    }

    @Override
    public void flush() throws IOException;

    @Override
    public void write(@PolySigned byte[] bytes, int offset, int length) throws IOException;

    @Override
    public void write(int oneByte) throws IOException;
}
