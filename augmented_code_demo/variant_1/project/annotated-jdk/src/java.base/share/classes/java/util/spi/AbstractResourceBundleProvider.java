/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.spi;

    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import jdk.internal.access.JavaUtilResourceBundleAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.util.resources.Bundles;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Locale;
    @Positive
import java.util.PropertyResourceBundle;
    @Positive
import java.util.ResourceBundle;
    @Positive
import static sun.security.util.SecurityConstants.GET_CLASSLOADER_PERMISSION;

    @Positive
public abstract class AbstractResourceBundleProvider implements ResourceBundleProvider {

    @Positive
    protected AbstractResourceBundleProvider() {
    @Positive
    }

    @Positive
    protected AbstractResourceBundleProvider(String... formats) {
    @Positive
    }

    @Positive
    protected String toBundleName(String baseName, Locale locale);

    @Positive
    @Override
    @Positive
    public ResourceBundle getBundle(@BinaryName String baseName, Locale locale);
    @Positive
}

// CFWR semantic augmentation - variant 1
