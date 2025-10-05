/*
    @Positive
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.FilterOutputStream;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.ByteArrayOutputStream;

    @Positive
@AnnotatedFor({ "mustcall", "signedness" })
    @Positive
public class DigestOutputStream extends FilterOutputStream {

    @Positive
    protected MessageDigest digest;

    @Positive
    @MustCallAlias
    @Positive
    public DigestOutputStream(@MustCallAlias OutputStream stream, MessageDigest digest) {
    @Positive
    }

    @Positive
    public MessageDigest getMessageDigest();

    @Positive
    public void setMessageDigest(MessageDigest digest);

    @Positive
    public void write(int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    public void on(boolean on);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
