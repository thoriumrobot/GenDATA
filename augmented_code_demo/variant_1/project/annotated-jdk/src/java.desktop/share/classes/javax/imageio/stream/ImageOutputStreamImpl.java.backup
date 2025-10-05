/*
 * Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.
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
package javax.imageio.stream;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.UTFDataFormatException;
import java.nio.ByteOrder;

@AnnotatedFor({ "signedness" })
public abstract class ImageOutputStreamImpl extends ImageInputStreamImpl implements ImageOutputStream {

    public ImageOutputStreamImpl() {
    }

    public abstract void write(int b) throws IOException;

    public void write(@PolySigned byte[] b) throws IOException;

    public abstract void write(@PolySigned byte[] b, int off, int len) throws IOException;

    public void writeBoolean(boolean v) throws IOException;

    public void writeByte(int v) throws IOException;

    public void writeShort(int v) throws IOException;

    public void writeChar(int v) throws IOException;

    public void writeInt(int v) throws IOException;

    public void writeLong(long v) throws IOException;

    public void writeFloat(float v) throws IOException;

    public void writeDouble(double v) throws IOException;

    public void writeBytes(String s) throws IOException;

    public void writeChars(String s) throws IOException;

    public void writeUTF(String s) throws IOException;

    public void writeShorts(short[] s, int off, int len) throws IOException;

    public void writeChars(char[] c, int off, int len) throws IOException;

    public void writeInts(int[] i, int off, int len) throws IOException;

    public void writeLongs(long[] l, int off, int len) throws IOException;

    public void writeFloats(float[] f, int off, int len) throws IOException;

    public void writeDoubles(double[] d, int off, int len) throws IOException;

    public void writeBit(int bit) throws IOException;

    public void writeBits(long bits, int numBits) throws IOException;

    protected final void flushBits() throws IOException;
}
