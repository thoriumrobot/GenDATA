/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.module;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.lang.module.ModuleReader;
    @Positive
import java.lang.module.ModuleReference;
    @Positive
import java.net.URI;
    @Positive
import java.util.Objects;
    @Positive
import java.util.function.Supplier;

    @Positive
public class ModuleReferenceImpl extends ModuleReference {

    @Positive
    public ModuleReferenceImpl(ModuleDescriptor descriptor, URI location, Supplier<ModuleReader> readerSupplier, ModulePatcher patcher, ModuleTarget target, ModuleHashes recordedHashes, ModuleHashes.HashSupplier hasher, ModuleResolution moduleResolution) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public ModuleReader open() throws IOException;

    @Positive
    public boolean isPatched();

    @Positive
    public ModuleTarget moduleTarget();

    @Positive
    public ModuleHashes recordedHashes();

    @Positive
    ModuleHashes.HashSupplier hasher();

    @Positive
    public ModuleResolution moduleResolution();

    @Positive
    public byte[] computeHash(String algorithm);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}
