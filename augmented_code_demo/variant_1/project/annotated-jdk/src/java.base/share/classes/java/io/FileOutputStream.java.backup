/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.channels.FileChannel;
import jdk.internal.access.SharedSecrets;
import jdk.internal.access.JavaIOFileDescriptorAccess;
import sun.nio.ch.FileChannelImpl;

@AnnotatedFor({ "index", "mustcall", "nullness", "signedness" })
public class FileOutputStream extends OutputStream {

    public FileOutputStream(String name) throws FileNotFoundException {
    }

    public FileOutputStream(String name, boolean append) throws FileNotFoundException {
    }

    public FileOutputStream(File file) throws FileNotFoundException {
    }

    public FileOutputStream(File file, boolean append) throws FileNotFoundException {
    }

    @MustCallAlias
    public FileOutputStream(@MustCallAlias FileDescriptor fdObj) {
    }

    public void write(@PolySigned int b) throws IOException;

    public void write(@PolySigned byte[] b) throws IOException;

    public void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    public void close() throws IOException;

    @MustCallAlias
    public final FileDescriptor getFD(@MustCallAlias FileOutputStream this) throws IOException;

    @MustCallAlias
    public FileChannel getChannel(@MustCallAlias FileOutputStream this);
}
