/*
 * Copyright (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util.zip;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.util.Arrays;
import sun.nio.cs.UTF_8;

@AnnotatedFor({ "index", "interning" })
@UsesObjectEquals
class ZipCoder {

    public static ZipCoder get(Charset charset);

    String toString(byte[] ba, int off, int length);

    String toString(byte[] ba, int length);

    String toString(byte[] ba);

    byte[] getBytes(String s);

    static String toStringUTF8(byte[] ba, int len);

    boolean isUTF8();

    int checkedHash(byte[] a, int off, int len) throws Exception;

    static int hash(String name);

    boolean hasTrailingSlash(byte[] a, int end);

    protected CharsetDecoder dec;

    protected CharsetDecoder decoder();

    static final class UTF8ZipCoder extends ZipCoder {

        @Override
        boolean isUTF8();

        @Override
        String toString(byte[] ba, int off, int length);

        @Override
        byte[] getBytes(String s);

        @Override
        int checkedHash(byte[] a, int off, int len) throws Exception;

        @Override
        boolean hasTrailingSlash(byte[] a, int end);
    }
}
