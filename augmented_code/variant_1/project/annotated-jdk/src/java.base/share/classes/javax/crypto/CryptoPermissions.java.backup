/*
 * Copyright (c) 1999, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.crypto;

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import java.security.*;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;
import java.io.Serializable;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.ObjectStreamField;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;
import static java.nio.charset.StandardCharsets.UTF_8;

final class CryptoPermissions extends PermissionCollection implements Serializable {

    void load(InputStream in) throws IOException, CryptoPolicyParser.ParsingException;

    @EnsuresNonEmptyIf(result = false, expression = "this")
    boolean isEmpty();

    @Override
    public void add(Permission permission);

    @Override
    public boolean implies(Permission permission);

    @Override
    public Enumeration<Permission> elements();

    CryptoPermissions getMinimum(CryptoPermissions other);

    PermissionCollection getPermissionCollection(String alg);
}

final class PermissionsEnumerator implements Enumeration<Permission> {

    @Override
    @EnsuresNonEmptyIf(result = true, expression = "this")
    public synchronized boolean hasMoreElements();

    @Override
    public synchronized Permission nextElement(@NonEmpty PermissionsEnumerator this);
}
