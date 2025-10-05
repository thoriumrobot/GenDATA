/*
    @Positive
 * Copyright (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jfr.internal.consumer;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.DataInput;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.RandomAccessFile;
    @Positive
import java.nio.file.Path;

    @Positive
public final class RecordingInput implements DataInput, AutoCloseable {

    @Positive
    private static final class Block {

    @Positive
        @Pure
    @Positive
        boolean contains(long position);

    @Positive
        public void read(RandomAccessFile file, int amount) throws IOException;

    @Positive
        public byte get(long position);

    @Positive
        public void reset();
    @Positive
    }

    @Positive
    public RecordingInput(File f, FileAccess fileAccess) throws IOException {
    @Positive
    }

    @Positive
    void positionPhysical(long position) throws IOException;

    @Positive
    byte readPhysicalByte() throws IOException;

    @Positive
    long readPhysicalLong() throws IOException;

    @Positive
    @Override
    @Positive
    public final byte readByte() throws IOException;

    @Positive
    @Override
    @Positive
    public final void readFully(byte[] dest, int offset, int length) throws IOException;

    @Positive
    @Override
    @Positive
    public final void readFully(byte[] dst) throws IOException;

    @Positive
    short readRawShort() throws IOException;

    @Positive
    @Override
    @Positive
    public double readDouble() throws IOException;

    @Positive
    @Override
    @Positive
    public float readFloat() throws IOException;

    @Positive
    int readRawInt() throws IOException;

    @Positive
    long readRawLong() throws IOException;

    @Positive
    public final long position();

    @Positive
    public final void position(long newPosition) throws IOException;

    @Positive
    long size();

    @Positive
    @Override
    @Positive
    public void close() throws IOException;

    @Positive
    @Override
    @Positive
    public final int skipBytes(int n) throws IOException;

    @Positive
    @Override
    @Positive
    public final boolean readBoolean() throws IOException;

    @Positive
    @Override
    @Positive
    public int readUnsignedByte() throws IOException;

    @Positive
    @Override
    @Positive
    public int readUnsignedShort() throws IOException;

    @Positive
    @Override
    @Positive
    public final String readLine() throws IOException;

    @Positive
    @Override
    @Positive
    public String readUTF() throws IOException;

    @Positive
    @Override
    @Positive
    public char readChar() throws IOException;

    @Positive
    @Override
    @Positive
    public short readShort() throws IOException;

    @Positive
    @Override
    @Positive
    public int readInt() throws IOException;

    @Positive
    @Override
    @Positive
    public long readLong() throws IOException;

    @Positive
    public void setValidSize(long size);

    @Positive
    public long getFileSize() throws IOException;

    @Positive
    public String getFilename();

    @Positive
    public void setFile(Path path) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
