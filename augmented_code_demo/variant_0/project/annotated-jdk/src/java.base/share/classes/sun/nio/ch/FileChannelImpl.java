/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.ch;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.ref.Cleaner.Cleanable;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.MappedByteBuffer;
    @Positive
import java.nio.channels.AsynchronousCloseException;
    @Positive
import java.nio.channels.ClosedByInterruptException;
    @Positive
import java.nio.channels.ClosedChannelException;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import java.nio.channels.FileLock;
    @Positive
import java.nio.channels.FileLockInterruptionException;
    @Positive
import java.nio.channels.NonReadableChannelException;
    @Positive
import java.nio.channels.NonWritableChannelException;
    @Positive
import java.nio.channels.ReadableByteChannel;
    @Positive
import java.nio.channels.SelectableChannel;
    @Positive
import java.nio.channels.WritableByteChannel;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.access.JavaIOFileDescriptorAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.ExtendedMapMode;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.misc.VM.BufferPool;
    @Positive
import jdk.internal.ref.Cleaner;
    @Positive
import jdk.internal.ref.CleanerFactory;
    @Positive
import jdk.internal.access.foreign.UnmapperProxy;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public class FileChannelImpl extends FileChannel {

    @Positive
    private static class Closer implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    public static FileChannel open(FileDescriptor fd, String path, boolean readable, boolean writable, boolean direct, Object parent);

    @Positive
    public void setUninterruptible();

    @Positive
    protected void implCloseChannel() throws IOException;

    @Positive
    public int read(ByteBuffer dst) throws IOException;

    @Positive
    public long read(ByteBuffer[] dsts, int offset, int length) throws IOException;

    @Positive
    public int write(ByteBuffer src) throws IOException;

    @Positive
    public long write(ByteBuffer[] srcs, int offset, int length) throws IOException;

    @Positive
    public long position() throws IOException;

    @Positive
    public FileChannel position(long newPosition) throws IOException;

    @Positive
    public long size() throws IOException;

    @Positive
    public FileChannel truncate(long newSize) throws IOException;

    @Positive
    public void force(boolean metaData) throws IOException;

    @Positive
    public long transferTo(long position, long count, WritableByteChannel target) throws IOException;

    @Positive
    public long transferFrom(ReadableByteChannel src, long position, long count) throws IOException;

    @Positive
    public int read(ByteBuffer dst, long position) throws IOException;

    @Positive
    public int write(ByteBuffer src, long position) throws IOException;

    @Positive
    private static abstract class Unmapper implements Runnable, UnmapperProxy {

    @Positive
        protected final long size;

    @Positive
        protected final long cap;

    @Positive
        @Override
    @Positive
        public long address();

    @Positive
        @Override
    @Positive
        public FileDescriptor fileDescriptor();

    @Positive
        @Override
    @Positive
        public void run();

    @Positive
        public void unmap();

    @Positive
        protected abstract void incrementStats();

    @Positive
        protected abstract void decrementStats();
    @Positive
    }

    @Positive
    private static class DefaultUnmapper extends Unmapper {

    @Positive
        public DefaultUnmapper(long address, long size, long cap, FileDescriptor fd, int pagePosition) {
    @Positive
        }

    @Positive
        protected void incrementStats();

    @Positive
        protected void decrementStats();

    @Positive
        public boolean isSync();
    @Positive
    }

    @Positive
    private static class SyncUnmapper extends Unmapper {

    @Positive
        public SyncUnmapper(long address, long size, long cap, FileDescriptor fd, int pagePosition) {
    @Positive
        }

    @Positive
        protected void incrementStats();

    @Positive
        protected void decrementStats();

    @Positive
        public boolean isSync();
    @Positive
    }

    @Positive
    public MappedByteBuffer map(MapMode mode, long position, long size) throws IOException;

    @Positive
    public Unmapper mapInternal(MapMode mode, long position, long size) throws IOException;

    @Positive
    public static BufferPool getMappedBufferPool();

    @Positive
    public static BufferPool getSyncMappedBufferPool();

    @Positive
    public FileLock lock(long position, long size, boolean shared) throws IOException;

    @Positive
    public FileLock tryLock(long position, long size, boolean shared) throws IOException;

    @Positive
    void release(FileLockImpl fli) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
