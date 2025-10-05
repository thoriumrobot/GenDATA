/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.cs;

import org.checkerframework.dataflow.qual.Pure;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CoderResult;
import java.util.Objects;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.vm.annotation.IntrinsicCandidate;

public class ISO_8859_1 extends Charset implements HistoricallyNamedCharset {

    public static final ISO_8859_1 INSTANCE;

    public ISO_8859_1() {
    }

    public String historicalName();

    @Pure
    public boolean contains(Charset cs);

    public CharsetDecoder newDecoder();

    public CharsetEncoder newEncoder();

    private static class Decoder extends CharsetDecoder {

        protected CoderResult decodeLoop(ByteBuffer src, CharBuffer dst);
    }

    private static class Encoder extends CharsetEncoder {

        public boolean canEncode(char c);

        public boolean isLegalReplacement(byte[] repl);

        protected CoderResult encodeLoop(CharBuffer src, ByteBuffer dst);
    }
}
