/*
    @Positive
 * Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
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
package javax.imageio.stream;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UTFDataFormatException;
    @Positive
import java.nio.ByteOrder;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public abstract class ImageOutputStreamImpl extends ImageInputStreamImpl implements ImageOutputStream {

    @Positive
    public ImageOutputStreamImpl() {
    @Positive
    }

    @Positive
    public abstract void write(int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b) throws IOException;

    @Positive
    public abstract void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    public void writeBoolean(boolean v) throws IOException;

    @Positive
    public void writeByte(int v) throws IOException;

    @Positive
    public void writeShort(int v) throws IOException;

    @Positive
    public void writeChar(int v) throws IOException;

    @Positive
    public void writeInt(int v) throws IOException;

    @Positive
    public void writeLong(long v) throws IOException;

    @Positive
    public void writeFloat(float v) throws IOException;

    @Positive
    public void writeDouble(double v) throws IOException;

    @Positive
    public void writeBytes(String s) throws IOException;

    @Positive
    public void writeChars(String s) throws IOException;

    @Positive
    public void writeUTF(String s) throws IOException;

    @Positive
    public void writeShorts(short[] s, int off, int len) throws IOException;

    @Positive
    public void writeChars(char[] c, int off, int len) throws IOException;

    @Positive
    public void writeInts(int[] i, int off, int len) throws IOException;

    @Positive
    public void writeLongs(long[] l, int off, int len) throws IOException;

    @Positive
    public void writeFloats(float[] f, int off, int len) throws IOException;

    @Positive
    public void writeDoubles(double[] d, int off, int len) throws IOException;

    @Positive
    public void writeBit(int bit) throws IOException;

    @Positive
    public void writeBits(long bits, int numBits) throws IOException;

    @Positive
    protected final void flushBits() throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
