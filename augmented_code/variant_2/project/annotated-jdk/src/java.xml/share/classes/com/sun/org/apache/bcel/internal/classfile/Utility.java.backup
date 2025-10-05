/*
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.bcel.internal.classfile;

import org.checkerframework.checker.signedness.qual.PolySigned;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.CharArrayReader;
import java.io.CharArrayWriter;
import java.io.FilterReader;
import java.io.FilterWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import com.sun.org.apache.bcel.internal.Const;
import com.sun.org.apache.bcel.internal.util.ByteSequence;

public abstract class Utility {

    public static String accessToString(final int access_flags);

    public static String accessToString(final int access_flags, final boolean for_class);

    public static String classOrInterface(final int access_flags);

    public static String codeToString(final byte[] code, final ConstantPool constant_pool, final int index, final int length, final boolean verbose);

    public static String codeToString(final byte[] code, final ConstantPool constant_pool, final int index, final int length);

    @SuppressWarnings("fallthrough")
    public static String codeToString(final ByteSequence bytes, final ConstantPool constant_pool, final boolean verbose) throws IOException;

    public static String codeToString(final ByteSequence bytes, final ConstantPool constant_pool) throws IOException;

    public static String compactClassName(final String str);

    public static String compactClassName(final String str, final boolean chopit);

    public static String compactClassName(String str, final String prefix, final boolean chopit);

    public static int setBit(final int flag, final int i);

    public static int clearBit(final int flag, final int i);

    public static boolean isSet(final int flag, final int i);

    public static String methodTypeToSignature(final String ret, final String[] argv) throws ClassFormatException;

    public static String[] methodSignatureArgumentTypes(final String signature) throws ClassFormatException;

    public static String[] methodSignatureArgumentTypes(final String signature, final boolean chopit) throws ClassFormatException;

    public static String methodSignatureReturnType(final String signature) throws ClassFormatException;

    public static String methodSignatureReturnType(final String signature, final boolean chopit) throws ClassFormatException;

    public static String methodSignatureToString(final String signature, final String name, final String access);

    public static String methodSignatureToString(final String signature, final String name, final String access, final boolean chopit);

    public static String methodSignatureToString(final String signature, final String name, final String access, final boolean chopit, final LocalVariableTable vars) throws ClassFormatException;

    public static String replace(String str, final String old, final String new_);

    public static String signatureToString(final String signature);

    public static String signatureToString(final String signature, final boolean chopit);

    public static String typeSignatureToString(final String signature, final boolean chopit) throws ClassFormatException;

    public static String getSignature(String type);

    public static byte typeOfMethodSignature(final String signature) throws ClassFormatException;

    public static byte typeOfSignature(final String signature) throws ClassFormatException;

    public static short searchOpcode(String name);

    public static String toHexString(@PolySigned final byte[] bytes);

    public static String format(final int i, final int length, final boolean left_justify, final char fill);

    public static String fillup(final String str, final int length, final boolean left_justify, final char fill);

    static boolean equals(final byte[] a, final byte[] b);

    public static void printArray(final PrintStream out, final Object[] obj);

    public static void printArray(final PrintWriter out, final Object[] obj);

    public static String printArray(final Object[] obj);

    public static String printArray(final Object[] obj, final boolean braces);

    public static String printArray(final Object[] obj, final boolean braces, final boolean quote);

    public static boolean isJavaIdentifierPart(final char ch);

    public static String encode(byte[] bytes, final boolean compress) throws IOException;

    public static byte[] decode(final String s, final boolean uncompress) throws IOException;

    private static class JavaReader extends FilterReader {

        public JavaReader(final Reader in) {
        }

        @Override
        public int read() throws IOException;

        @Override
        public int read(final char[] cbuf, final int off, final int len) throws IOException;
    }

    private static class JavaWriter extends FilterWriter {

        public JavaWriter(final Writer out) {
        }

        @Override
        public void write(final int b) throws IOException;

        @Override
        public void write(final char[] cbuf, final int off, final int len) throws IOException;

        @Override
        public void write(final String str, final int off, final int len) throws IOException;
    }

    public static String convertString(final String label);
}
