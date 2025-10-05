/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.crypto;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.*;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Vector;
    @Positive
import static java.util.Locale.ENGLISH;
    @Positive
import java.security.GeneralSecurityException;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import java.lang.reflect.*;

    @Positive
final class CryptoPolicyParser {

    @Positive
    void read(Reader policy) throws ParsingException, IOException;

    @Positive
    CryptoPermission[] getPermissions();

    @Positive
    private static class GrantEntry {

    @Positive
        void add(CryptoPermissionEntry pe);

    @Positive
        boolean remove(CryptoPermissionEntry pe);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        boolean contains(CryptoPermissionEntry pe);

    @Positive
        Enumeration<CryptoPermissionEntry> permissionElements();
    @Positive
    }

    @Positive
    private static class CryptoPermissionEntry {

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    static final class ParsingException extends GeneralSecurityException {
    @Positive
    }
    @Positive
}
