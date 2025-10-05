/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.utils;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FilterOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public class UnsyncBufferedOutputStream extends FilterOutputStream {

    @Positive
    protected byte[] buffer;

    @Positive
    protected int count;

    @Positive
    public UnsyncBufferedOutputStream(OutputStream out) {
    @Positive
    }

    @Positive
    public UnsyncBufferedOutputStream(OutputStream out, int size) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void flush() throws IOException;

    @Positive
    @Override
    @Positive
    public void write(@PolySigned byte[] bytes, int offset, int length) throws IOException;

    @Positive
    @Override
    @Positive
    public void write(int oneByte) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
