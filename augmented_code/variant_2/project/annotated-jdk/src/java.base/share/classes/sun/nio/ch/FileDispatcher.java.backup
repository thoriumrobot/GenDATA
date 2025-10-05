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
package sun.nio.ch;

import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FileDescriptor;
import java.io.IOException;
import java.nio.channels.SelectableChannel;

@AnnotatedFor({ "index" })
abstract class FileDispatcher extends NativeDispatcher {

    public static final int NO_LOCK;

    public static final int LOCKED;

    public static final int RET_EX_LOCK;

    public static final int INTERRUPTED;

    abstract long seek(FileDescriptor fd, long offset) throws IOException;

    abstract int force(FileDescriptor fd, boolean metaData) throws IOException;

    abstract int truncate(FileDescriptor fd, long size) throws IOException;

    abstract long size(FileDescriptor fd) throws IOException;

    abstract int lock(FileDescriptor fd, boolean blocking, long pos, long size, boolean shared) throws IOException;

    abstract void release(FileDescriptor fd, long pos, long size) throws IOException;

    abstract FileDescriptor duplicateForMapping(FileDescriptor fd) throws IOException;

    abstract boolean canTransferToDirectly(SelectableChannel sc);

    abstract boolean transferToDirectlyNeedsPositionLock();

    abstract int setDirectIO(FileDescriptor fd, String path);
}
