/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import java.nio.channels.SelectableChannel;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
abstract class FileDispatcher extends NativeDispatcher {

    @Positive
    public static final int NO_LOCK;

    @Positive
    public static final int LOCKED;

    @Positive
    public static final int RET_EX_LOCK;

    @Positive
    public static final int INTERRUPTED;

    @Positive
    abstract long seek(FileDescriptor fd, long offset) throws IOException;

    @Positive
    abstract int force(FileDescriptor fd, boolean metaData) throws IOException;

    @Positive
    abstract int truncate(FileDescriptor fd, long size) throws IOException;

    @Positive
    abstract long size(FileDescriptor fd) throws IOException;

    @Positive
    abstract int lock(FileDescriptor fd, boolean blocking, long pos, long size, boolean shared) throws IOException;

    @Positive
    abstract void release(FileDescriptor fd, long pos, long size) throws IOException;

    @Positive
    abstract FileDescriptor duplicateForMapping(FileDescriptor fd) throws IOException;

    @Positive
    abstract boolean canTransferToDirectly(SelectableChannel sc);

    @Positive
    abstract boolean transferToDirectlyNeedsPositionLock();

    @Positive
    abstract int setDirectIO(FileDescriptor fd, String path);
    @Positive
}
