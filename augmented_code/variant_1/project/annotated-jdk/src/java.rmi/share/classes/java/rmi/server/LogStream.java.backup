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
package java.rmi.server;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.util.*;

@AnnotatedFor({ "signedness" })
@Deprecated
public class LogStream extends PrintStream {

    @Deprecated
    public static LogStream log(String name);

    @Deprecated
    public static synchronized PrintStream getDefaultStream();

    @Deprecated
    public static synchronized void setDefaultStream(PrintStream newDefault);

    @Deprecated
    public synchronized OutputStream getOutputStream();

    @Deprecated
    public synchronized void setOutputStream(OutputStream out);

    @Deprecated
    public void write(int b);

    @Deprecated
    public void write(@PolySigned byte[] b, int off, int len);

    @Deprecated
    public String toString();

    public static final int SILENT;

    public static final int BRIEF;

    public static final int VERBOSE;

    @Deprecated
    public static int parseLevel(String s);
}
