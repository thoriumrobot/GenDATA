/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.cs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
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
import java.nio.charset.CoderResult;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
public class ISO_8859_1 extends Charset implements HistoricallyNamedCharset {

    @Positive
    public static final ISO_8859_1 INSTANCE;

    @Positive
    public ISO_8859_1() {
    @Positive
    }

    @Positive
    public String historicalName();

    @Positive
    @Pure
    @Positive
    public boolean contains(Charset cs);

    @Positive
    public CharsetDecoder newDecoder();

    @Positive
    public CharsetEncoder newEncoder();

    @Positive
    private static class Decoder extends CharsetDecoder {

    @Positive
        protected CoderResult decodeLoop(ByteBuffer src, CharBuffer dst);
    @Positive
    }

    @Positive
    private static class Encoder extends CharsetEncoder {

    @Positive
        public boolean canEncode(char c);

    @Positive
        public boolean isLegalReplacement(byte[] repl);

    @Positive
        protected CoderResult encodeLoop(CharBuffer src, ByteBuffer dst);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
