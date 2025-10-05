/*
    @Positive
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FilterOutputStream;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import sun.nio.cs.ISO_8859_1;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public class Base64 {

    @Positive
    @Pure
    @Positive
    public static Encoder getEncoder();

    @Positive
    @Pure
    @Positive
    public static Encoder getUrlEncoder();

    @Positive
    @Pure
    @Positive
    public static Encoder getMimeEncoder();

    @Positive
    public static Encoder getMimeEncoder(int lineLength, byte[] lineSeparator);

    @Positive
    @Pure
    @Positive
    public static Decoder getDecoder();

    @Positive
    @Pure
    @Positive
    public static Decoder getUrlDecoder();

    @Positive
    @Pure
    @Positive
    public static Decoder getMimeDecoder();

    @Positive
    public static class Encoder {

    @Positive
        public byte[] encode(byte[] src);

    @Positive
        public int encode(byte[] src, byte[] dst);

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public String encodeToString(@PolySigned byte[] src);

    @Positive
        public ByteBuffer encode(ByteBuffer buffer);

    @Positive
        public OutputStream wrap(OutputStream os);

    @Positive
        public Encoder withoutPadding();
    @Positive
    }

    @Positive
    public static class Decoder {

    @Positive
        public byte[] decode(byte[] src);

    @Positive
        @PolySigned
    @Positive
        public byte[] decode(String src);

    @Positive
        public int decode(byte[] src, byte[] dst);

    @Positive
        public ByteBuffer decode(ByteBuffer buffer);

    @Positive
        public InputStream wrap(InputStream is);
    @Positive
    }

    @Positive
    private static class EncOutputStream extends FilterOutputStream {

    @Positive
        @Override
    @Positive
        public void write(int b) throws IOException;

    @Positive
        @Override
    @Positive
        public void write(byte[] b, int off, int len) throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;
    @Positive
    }

    @Positive
    private static class DecInputStream extends InputStream {

    @Positive
        @Override
    @Positive
        public int read() throws IOException;

    @Positive
        @Override
    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        @Override
    @Positive
        public int available() throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
