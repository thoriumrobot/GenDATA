/*
 * Copyright (c) 2007, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.file.*;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.spi.*;
import java.nio.ByteBuffer;
import java.io.IOException;
import java.util.concurrent.Future;
import java.util.concurrent.ExecutorService;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AsynchronousFileChannel implements AsynchronousChannel {

    protected AsynchronousFileChannel() {
    }

    public static AsynchronousFileChannel open(Path file, Set<? extends OpenOption> options, ExecutorService executor, FileAttribute<?>... attrs) throws IOException;

    public static AsynchronousFileChannel open(Path file, OpenOption... options) throws IOException;

    public abstract long size() throws IOException;

    public abstract AsynchronousFileChannel truncate(long size) throws IOException;

    public abstract void force(boolean metaData) throws IOException;

    public abstract <A> void lock(long position, long size, boolean shared, A attachment, CompletionHandler<FileLock, ? super A> handler);

    public final <A> void lock(A attachment, CompletionHandler<FileLock, ? super A> handler);

    public abstract Future<FileLock> lock(long position, long size, boolean shared);

    public final Future<FileLock> lock();

    public abstract FileLock tryLock(long position, long size, boolean shared) throws IOException;

    public final FileLock tryLock() throws IOException;

    public abstract <A> void read(ByteBuffer dst, long position, A attachment, CompletionHandler<Integer, ? super A> handler);

    public abstract Future<Integer> read(ByteBuffer dst, long position);

    public abstract <A> void write(ByteBuffer src, long position, A attachment, CompletionHandler<Integer, ? super A> handler);

    public abstract Future<Integer> write(ByteBuffer src, long position);
}
