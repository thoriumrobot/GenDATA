/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.net.URL;
import java.net.SocketPermission;
import java.util.ArrayList;
import java.util.List;
import java.util.Hashtable;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.security.cert.*;
import java.util.Objects;
import sun.net.util.URLUtil;
import sun.security.util.IOUtils;

public class CodeSource implements java.io.Serializable {

    public CodeSource(URL url, java.security.cert.Certificate[] certs) {
    }

    public CodeSource(URL url, CodeSigner[] signers) {
    }

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public final URL getLocation();

    String getLocationNoFragString();

    public final java.security.cert.Certificate[] getCertificates();

    public final CodeSigner[] getCodeSigners();

    public boolean implies(CodeSource codesource);

    boolean matchCerts(CodeSource that, boolean strict);

    @Override
    public String toString();
}
