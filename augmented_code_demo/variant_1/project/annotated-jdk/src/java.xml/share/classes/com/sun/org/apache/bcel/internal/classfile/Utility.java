/*
    @Positive
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.bcel.internal.classfile;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.CharArrayReader;
    @Positive
import java.io.CharArrayWriter;
    @Positive
import java.io.FilterReader;
    @Positive
import java.io.FilterWriter;
    @Positive
import java.io.IOException;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Writer;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.zip.GZIPInputStream;
    @Positive
import java.util.zip.GZIPOutputStream;
    @Positive
import com.sun.org.apache.bcel.internal.Const;
    @Positive
import com.sun.org.apache.bcel.internal.util.ByteSequence;

    @Positive
public abstract class Utility {

    @Positive
    public static String accessToString(final int access_flags);

    @Positive
    public static String accessToString(final int access_flags, final boolean for_class);

    @Positive
    public static String classOrInterface(final int access_flags);

    @Positive
    public static String codeToString(final byte[] code, final ConstantPool constant_pool, final int index, final int length, final boolean verbose);

    @Positive
    public static String codeToString(final byte[] code, final ConstantPool constant_pool, final int index, final int length);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public static String codeToString(final ByteSequence bytes, final ConstantPool constant_pool, final boolean verbose) throws IOException;

    @Positive
    public static String codeToString(final ByteSequence bytes, final ConstantPool constant_pool) throws IOException;

    @Positive
    public static String compactClassName(final String str);

    @Positive
    public static String compactClassName(final String str, final boolean chopit);

    @Positive
    public static String compactClassName(String str, final String prefix, final boolean chopit);

    @Positive
    public static int setBit(final int flag, final int i);

    @Positive
    public static int clearBit(final int flag, final int i);

    @Positive
    public static boolean isSet(final int flag, final int i);

    @Positive
    public static String methodTypeToSignature(final String ret, final String[] argv) throws ClassFormatException;

    @Positive
    public static String[] methodSignatureArgumentTypes(final String signature) throws ClassFormatException;

    @Positive
    public static String[] methodSignatureArgumentTypes(final String signature, final boolean chopit) throws ClassFormatException;

    @Positive
    public static String methodSignatureReturnType(final String signature) throws ClassFormatException;

    @Positive
    public static String methodSignatureReturnType(final String signature, final boolean chopit) throws ClassFormatException;

    @Positive
    public static String methodSignatureToString(final String signature, final String name, final String access);

    @Positive
    public static String methodSignatureToString(final String signature, final String name, final String access, final boolean chopit);

    @Positive
    public static String methodSignatureToString(final String signature, final String name, final String access, final boolean chopit, final LocalVariableTable vars) throws ClassFormatException;

    @Positive
    public static String replace(String str, final String old, final String new_);

    @Positive
    public static String signatureToString(final String signature);

    @Positive
    public static String signatureToString(final String signature, final boolean chopit);

    @Positive
    public static String typeSignatureToString(final String signature, final boolean chopit) throws ClassFormatException;

    @Positive
    public static String getSignature(String type);

    @Positive
    public static byte typeOfMethodSignature(final String signature) throws ClassFormatException;

    @Positive
    public static byte typeOfSignature(final String signature) throws ClassFormatException;

    @Positive
    public static short searchOpcode(String name);

    @Positive
    public static String toHexString(@PolySigned final byte[] bytes);

    @Positive
    public static String format(final int i, final int length, final boolean left_justify, final char fill);

    @Positive
    public static String fillup(final String str, final int length, final boolean left_justify, final char fill);

    @Positive
    static boolean equals(final byte[] a, final byte[] b);

    @Positive
    public static void printArray(final PrintStream out, final Object[] obj);

    @Positive
    public static void printArray(final PrintWriter out, final Object[] obj);

    @Positive
    public static String printArray(final Object[] obj);

    @Positive
    public static String printArray(final Object[] obj, final boolean braces);

    @Positive
    public static String printArray(final Object[] obj, final boolean braces, final boolean quote);

    @Positive
    public static boolean isJavaIdentifierPart(final char ch);

    @Positive
    public static String encode(byte[] bytes, final boolean compress) throws IOException;

    @Positive
    public static byte[] decode(final String s, final boolean uncompress) throws IOException;

    @Positive
    private static class JavaReader extends FilterReader {

    @Positive
        public JavaReader(final Reader in) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int read() throws IOException;

    @Positive
        @Override
    @Positive
        public int read(final char[] cbuf, final int off, final int len) throws IOException;
    @Positive
    }

    @Positive
    private static class JavaWriter extends FilterWriter {

    @Positive
        public JavaWriter(final Writer out) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void write(final int b) throws IOException;

    @Positive
        @Override
    @Positive
        public void write(final char[] cbuf, final int off, final int len) throws IOException;

    @Positive
        @Override
    @Positive
        public void write(final String str, final int off, final int len) throws IOException;
    @Positive
    }

    @Positive
    public static String convertString(final String label);
    @Positive
}

// CFWR semantic augmentation - variant 1
