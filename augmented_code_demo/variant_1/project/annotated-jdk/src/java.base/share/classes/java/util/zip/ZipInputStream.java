/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.PushbackInputStream;
    @Positive
import java.nio.charset.Charset;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import static java.util.zip.ZipConstants64.*;
    @Positive
import static java.util.zip.ZipUtils.*;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public class ZipInputStream extends InflaterInputStream implements ZipConstants {

    @Positive
    @MustCallAlias
    @Positive
    public ZipInputStream(@MustCallAlias InputStream in) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public ZipInputStream(@MustCallAlias InputStream in, Charset charset) {
    @Positive
    }

    @Positive
    public ZipEntry getNextEntry() throws IOException;

    @Positive
    public void closeEntry() throws IOException;

    @Positive
    public int available() throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public int read(byte[] b, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws IOException;

    @Positive
    public long skip(long n) throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    protected ZipEntry createZipEntry(String name);
    @Positive
}

// CFWR semantic augmentation - variant 1
