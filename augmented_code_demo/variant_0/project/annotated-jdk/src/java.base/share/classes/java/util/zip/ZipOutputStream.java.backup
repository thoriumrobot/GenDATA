/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.OutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Vector;
import java.util.HashSet;
import static java.util.zip.ZipConstants64.*;
import static java.util.zip.ZipUtils.*;
import sun.nio.cs.UTF_8;
import sun.security.action.GetPropertyAction;

@AnnotatedFor({ "index", "signedness" })
public class ZipOutputStream extends DeflaterOutputStream implements ZipConstants {

    private static class XEntry {

        public XEntry(ZipEntry entry, long offset) {
        }
    }

    public static final int STORED;

    public static final int DEFLATED;

    @MustCallAlias
    public ZipOutputStream(@MustCallAlias OutputStream out) {
    }

    @MustCallAlias
    public ZipOutputStream(@MustCallAlias OutputStream out, Charset charset) {
    }

    public void setComment(String comment);

    public void setMethod(int method);

    public void setLevel(int level);

    public void putNextEntry(ZipEntry e) throws IOException;

    public void closeEntry() throws IOException;

    public synchronized void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws IOException;

    public void finish() throws IOException;

    public void close() throws IOException;
}
