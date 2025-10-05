/*
    @Positive
 * Copyright (c) 2000, 2012, Oracle and/or its affiliates. All rights reserved.
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
package javax.imageio.stream;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.RandomAccessFile;
    @Positive
import java.nio.file.Files;
    @Positive
import com.sun.imageio.stream.StreamCloser;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
public class FileCacheImageOutputStream extends ImageOutputStreamImpl {

    @Positive
    public FileCacheImageOutputStream(OutputStream stream, File cacheDir) throws IOException {
    @Positive
    }

    @Positive
    public int read() throws IOException;

    @Positive
    public int read(byte[] b, int off, int len) throws IOException;

    @Positive
    public void write(int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    public long length();

    @Positive
    public void seek(long pos) throws IOException;

    @Positive
    public boolean isCached();

    @Positive
    public boolean isCachedFile();

    @Positive
    public boolean isCachedMemory();

    @Positive
    public void close() throws IOException;

    @Positive
    public void flushBefore(long pos) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
