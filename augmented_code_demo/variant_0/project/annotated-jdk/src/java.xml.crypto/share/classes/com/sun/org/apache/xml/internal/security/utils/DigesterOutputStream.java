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
import java.io.ByteArrayOutputStream;
    @Positive
import com.sun.org.apache.xml.internal.security.algorithms.MessageDigestAlgorithm;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public class DigesterOutputStream extends ByteArrayOutputStream {

    @Positive
    public DigesterOutputStream(MessageDigestAlgorithm mda) {
    @Positive
    }

    @Positive
    public void write(@PolySigned byte[] arg0);

    @Positive
    public void write(int arg0);

    @Positive
    public void write(@PolySigned byte[] arg0, int arg1, int arg2);

    @Positive
    public byte[] getDigestValue();
    @Positive
}

// CFWR semantic augmentation - variant 0
