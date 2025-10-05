/*
 * Copyright (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jfr.internal.consumer;

import org.checkerframework.dataflow.qual.Pure;
import java.io.DataInput;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;

public final class RecordingInput implements DataInput, AutoCloseable {

    private static final class Block {

        @Pure
        boolean contains(long position);

        public void read(RandomAccessFile file, int amount) throws IOException;

        public byte get(long position);

        public void reset();
    }

    public RecordingInput(File f, FileAccess fileAccess) throws IOException {
    }

    void positionPhysical(long position) throws IOException;

    byte readPhysicalByte() throws IOException;

    long readPhysicalLong() throws IOException;

    @Override
    public final byte readByte() throws IOException;

    @Override
    public final void readFully(byte[] dest, int offset, int length) throws IOException;

    @Override
    public final void readFully(byte[] dst) throws IOException;

    short readRawShort() throws IOException;

    @Override
    public double readDouble() throws IOException;

    @Override
    public float readFloat() throws IOException;

    int readRawInt() throws IOException;

    long readRawLong() throws IOException;

    public final long position();

    public final void position(long newPosition) throws IOException;

    long size();

    @Override
    public void close() throws IOException;

    @Override
    public final int skipBytes(int n) throws IOException;

    @Override
    public final boolean readBoolean() throws IOException;

    @Override
    public int readUnsignedByte() throws IOException;

    @Override
    public int readUnsignedShort() throws IOException;

    @Override
    public final String readLine() throws IOException;

    @Override
    public String readUTF() throws IOException;

    @Override
    public char readChar() throws IOException;

    @Override
    public short readShort() throws IOException;

    @Override
    public int readInt() throws IOException;

    @Override
    public long readLong() throws IOException;

    public void setValidSize(long size);

    public long getFileSize() throws IOException;

    public String getFilename();

    public void setFile(Path path) throws IOException;
}
