/*
    @Positive
 * Copyright (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util.zip;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.CharacterCodingException;
    @Positive
import java.nio.charset.CodingErrorAction;
    @Positive
import java.util.Arrays;
    @Positive
import sun.nio.cs.UTF_8;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class ZipCoder {

    @Positive
    public static ZipCoder get(Charset charset);

    @Positive
    String toString(byte[] ba, int off, int length);

    @Positive
    String toString(byte[] ba, int length);

    @Positive
    String toString(byte[] ba);

    @Positive
    byte[] getBytes(String s);

    @Positive
    static String toStringUTF8(byte[] ba, int len);

    @Positive
    boolean isUTF8();

    @Positive
    int checkedHash(byte[] a, int off, int len) throws Exception;

    @Positive
    static int hash(String name);

    @Positive
    boolean hasTrailingSlash(byte[] a, int end);

    @Positive
    protected CharsetDecoder dec;

    @Positive
    protected CharsetDecoder decoder();

    @Positive
    static final class UTF8ZipCoder extends ZipCoder {

    @Positive
        @Override
    @Positive
        boolean isUTF8();

    @Positive
        @Override
    @Positive
        String toString(byte[] ba, int off, int length);

    @Positive
        @Override
    @Positive
        byte[] getBytes(String s);

    @Positive
        @Override
    @Positive
        int checkedHash(byte[] a, int off, int len) throws Exception;

    @Positive
        @Override
    @Positive
        boolean hasTrailingSlash(byte[] a, int end);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
