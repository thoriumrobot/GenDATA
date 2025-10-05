/*
    @Positive
 * Copyright (c) 2007, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.file.*;
    @Positive
import java.nio.file.attribute.FileAttribute;
    @Positive
import java.nio.file.spi.*;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.io.IOException;
    @Positive
import java.util.concurrent.Future;
    @Positive
import java.util.concurrent.ExecutorService;
    @Positive
import java.util.Set;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Collections;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class AsynchronousFileChannel implements AsynchronousChannel {

    @Positive
    protected AsynchronousFileChannel() {
    @Positive
    }

    @Positive
    public static AsynchronousFileChannel open(Path file, Set<? extends OpenOption> options, ExecutorService executor, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public static AsynchronousFileChannel open(Path file, OpenOption... options) throws IOException;

    @Positive
    public abstract long size() throws IOException;

    @Positive
    public abstract AsynchronousFileChannel truncate(long size) throws IOException;

    @Positive
    public abstract void force(boolean metaData) throws IOException;

    @Positive
    public abstract <A> void lock(long position, long size, boolean shared, A attachment, CompletionHandler<FileLock, ? super A> handler);

    @Positive
    public final <A> void lock(A attachment, CompletionHandler<FileLock, ? super A> handler);

    @Positive
    public abstract Future<FileLock> lock(long position, long size, boolean shared);

    @Positive
    public final Future<FileLock> lock();

    @Positive
    public abstract FileLock tryLock(long position, long size, boolean shared) throws IOException;

    @Positive
    public final FileLock tryLock() throws IOException;

    @Positive
    public abstract <A> void read(ByteBuffer dst, long position, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Positive
    public abstract Future<Integer> read(ByteBuffer dst, long position);

    @Positive
    public abstract <A> void write(ByteBuffer src, long position, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Positive
    public abstract Future<Integer> write(ByteBuffer src, long position);
    @Positive
}

// CFWR semantic augmentation - variant 1
