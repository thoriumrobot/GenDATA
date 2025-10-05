/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package sun.nio.cs.ext;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CoderResult;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Arrays;
    @Positive
import sun.nio.cs.CharsetMapping;
    @Positive
import sun.nio.cs.*;

    @Positive
public class SJIS_0213 extends Charset {

    @Positive
    public SJIS_0213() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean contains(Charset cs);

    @Positive
    public CharsetDecoder newDecoder();

    @Positive
    public CharsetEncoder newEncoder();

    @Positive
    private static class Holder {
    @Positive
    }

    @Positive
    protected static class Decoder extends CharsetDecoder {

    @Positive
        protected static final char UNMAPPABLE;

    @Positive
        protected Decoder(Charset cs) {
    @Positive
        }

    @Positive
        protected CoderResult decodeLoop(ByteBuffer src, CharBuffer dst);

    @Positive
        protected char decodeSingle(int b);

    @Positive
        protected char decodeDouble(int b1, int b2);

    @Positive
        protected char[] decodeDoubleEx(int b1, int b2);
    @Positive
    }

    @Positive
    protected static class Encoder extends CharsetEncoder {

    @Positive
        protected static final int UNMAPPABLE;

    @Positive
        protected static final int MAX_SINGLEBYTE;

    @Positive
        protected Encoder(Charset cs) {
    @Positive
        }

    @Positive
        public boolean canEncode(char c);

    @Positive
        protected int encodeChar(char ch);

    @Positive
        protected int encodeSurrogate(char hi, char lo);

    @Positive
        protected int encodeComposite(char base, char cc);

    @Positive
        protected boolean isCompositeBase(char ch);

    @Positive
        protected CoderResult encodeArrayLoop(CharBuffer src, ByteBuffer dst);

    @Positive
        protected CoderResult encodeBufferLoop(CharBuffer src, ByteBuffer dst);

    @Positive
        protected CoderResult encodeLoop(CharBuffer src, ByteBuffer dst);

    @Positive
        protected CoderResult implFlush(ByteBuffer dst);

    @Positive
        protected void implReset();
    @Positive
    }
    @Positive
}
