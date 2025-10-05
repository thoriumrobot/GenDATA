/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package java.nio.channels;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.mustcall.qual.NotOwning;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.MappedByteBuffer;
    @Positive
import java.nio.channels.spi.AbstractInterruptibleChannel;
    @Positive
import java.nio.file.*;
    @Positive
import java.nio.file.attribute.FileAttribute;
    @Positive
import java.nio.file.spi.*;
    @Positive
import java.util.Set;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Collections;

    @Positive
@AnnotatedFor({ "index", "mustcall", "nullness" })
    @Positive
public abstract class FileChannel extends AbstractInterruptibleChannel implements SeekableByteChannel, GatheringByteChannel, ScatteringByteChannel {

    @Positive
    protected FileChannel() {
    @Positive
    }

    @Positive
    public static FileChannel open(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public static FileChannel open(Path path, OpenOption... options) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int read(ByteBuffer dst) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    public abstract long read(ByteBuffer[] dsts, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int length) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    public final long read(ByteBuffer[] dsts) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract int write(ByteBuffer src) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract long write(ByteBuffer[] srcs, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int length) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public final long write(ByteBuffer[] srcs) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract long position() throws IOException;

    @Positive
    @NotOwning
    @Positive
    public abstract FileChannel position(@NonNegative long newPosition) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract long size() throws IOException;

    @Positive
    @NotOwning
    @Positive
    public abstract FileChannel truncate(@NonNegative long size) throws IOException;

    @Positive
    public abstract void force(boolean metaData) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract long transferTo(@NonNegative long position, @NonNegative long count, WritableByteChannel target) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract long transferFrom(ReadableByteChannel src, @NonNegative long position, @NonNegative long count) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    public abstract int read(ByteBuffer dst, @NonNegative long position) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public abstract int write(ByteBuffer src, @NonNegative long position) throws IOException;

    @Positive
    public static class MapMode {

    @Positive
        public static final MapMode READ_ONLY;

    @Positive
        public static final MapMode READ_WRITE;

    @Positive
        public static final MapMode PRIVATE;

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public abstract MappedByteBuffer map(MapMode mode, @NonNegative long position, @NonNegative long size) throws IOException;

    @Positive
    public abstract FileLock lock(@NonNegative long position, @NonNegative long size, boolean shared) throws IOException;

    @Positive
    public final FileLock lock() throws IOException;

    @Positive
    @Nullable
    @Positive
    public abstract FileLock tryLock(@NonNegative long position, @NonNegative long size, boolean shared) throws IOException;

    @Positive
    @Nullable
    @Positive
    public final FileLock tryLock() throws IOException;
    @Positive
}
