/*
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.cs.ext;

import org.checkerframework.dataflow.qual.Pure;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Arrays;
import sun.nio.cs.CharsetMapping;
import sun.nio.cs.*;

public class SJIS_0213 extends Charset {

    public SJIS_0213() {
    }

    @Pure
    public boolean contains(Charset cs);

    public CharsetDecoder newDecoder();

    public CharsetEncoder newEncoder();

    private static class Holder {
    }

    protected static class Decoder extends CharsetDecoder {

        protected static final char UNMAPPABLE;

        protected Decoder(Charset cs) {
        }

        protected CoderResult decodeLoop(ByteBuffer src, CharBuffer dst);

        protected char decodeSingle(int b);

        protected char decodeDouble(int b1, int b2);

        protected char[] decodeDoubleEx(int b1, int b2);
    }

    protected static class Encoder extends CharsetEncoder {

        protected static final int UNMAPPABLE;

        protected static final int MAX_SINGLEBYTE;

        protected Encoder(Charset cs) {
        }

        public boolean canEncode(char c);

        protected int encodeChar(char ch);

        protected int encodeSurrogate(char hi, char lo);

        protected int encodeComposite(char base, char cc);

        protected boolean isCompositeBase(char ch);

        protected CoderResult encodeArrayLoop(CharBuffer src, ByteBuffer dst);

        protected CoderResult encodeBufferLoop(CharBuffer src, ByteBuffer dst);

        protected CoderResult encodeLoop(CharBuffer src, ByteBuffer dst);

        protected CoderResult implFlush(ByteBuffer dst);

        protected void implReset();
    }
}
