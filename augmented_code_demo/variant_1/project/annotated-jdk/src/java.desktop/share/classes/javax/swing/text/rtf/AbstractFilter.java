/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.text.rtf;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.lang.*;

    @Positive
@AnnotatedFor("signedness")
    @Positive
abstract class AbstractFilter extends OutputStream {

    @Positive
    protected char[] translationTable;

    @Positive
    protected boolean[] specialsTable;

    @Positive
    public void readFromStream(InputStream in) throws IOException;

    @Positive
    public void readFromReader(Reader in) throws IOException;

    @Positive
    public AbstractFilter() {
    @Positive
    }

    @Positive
    public void write(int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] buf, int off, int len) throws IOException;

    @Positive
    public void write(String s) throws IOException;

    @Positive
    protected abstract void write(char ch) throws IOException;

    @Positive
    protected abstract void writeSpecial(int b) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
