/*
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FilterInputStream;
import java.io.InputStream;
import java.io.IOException;

@AnnotatedFor({ "index" })
public class DeflaterInputStream extends FilterInputStream {

    protected final Deflater def;

    protected final byte[] buf;

    @MustCallAlias
    public DeflaterInputStream(@MustCallAlias InputStream in) {
    }

    @MustCallAlias
    public DeflaterInputStream(@MustCallAlias InputStream in, Deflater defl) {
    }

    @MustCallAlias
    public DeflaterInputStream(@MustCallAlias InputStream in, Deflater defl, @Positive int bufLen) {
    }

    public void close() throws IOException;

    public int read() throws IOException;

    @GTENegativeOne
    @LTEqLengthOf({ "#1" })
    public int read(byte[] b, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws IOException;

    public long skip(long n) throws IOException;

    public int available() throws IOException;

    public boolean markSupported();

    public void mark(@NonNegative int limit);

    public void reset() throws IOException;
}
