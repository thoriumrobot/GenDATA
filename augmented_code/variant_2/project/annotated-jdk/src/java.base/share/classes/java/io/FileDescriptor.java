/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.access.JavaIOFileDescriptorAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.ref.PhantomCleanable;

    @Positive
@AnnotatedFor({ "nullness", "index" })
    @Positive
public final class FileDescriptor {

    @Positive
    public FileDescriptor() {
    @Positive
    }

    @Positive
    public static final FileDescriptor in;

    @Positive
    public static final FileDescriptor out;

    @Positive
    public static final FileDescriptor err;

    @Positive
    public boolean valid();

    @Positive
    public native void sync() throws SyncFailedException;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    synchronized void set(int fd);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    void setHandle(long handle);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    synchronized void registerCleanup(PhantomCleanable<FileDescriptor> cleanable);

    @Positive
    synchronized void unregisterCleanup();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    synchronized void close() throws IOException;

    @Positive
    synchronized void attach(Closeable c);

    @Positive
    @SuppressWarnings("try")
    @Positive
    synchronized void closeAll(Closeable releaser) throws IOException;
    @Positive
}
