/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.access.JavaIOFileDescriptorAccess;
    @Positive
import sun.nio.ch.FileChannelImpl;

    @Positive
@AnnotatedFor({ "index", "mustcall", "nullness", "signedness" })
    @Positive
public class FileOutputStream extends OutputStream {

    @Positive
    public FileOutputStream(String name) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public FileOutputStream(String name, boolean append) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public FileOutputStream(File file) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public FileOutputStream(File file, boolean append) throws FileNotFoundException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public FileOutputStream(@MustCallAlias FileDescriptor fdObj) {
    @Positive
    }

    @Positive
    public void write(@PolySigned int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public final FileDescriptor getFD(@MustCallAlias FileOutputStream this) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public FileChannel getChannel(@MustCallAlias FileOutputStream this);
    @Positive
}
