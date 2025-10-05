/*
    @Positive
 * Copyright (c) 2015, 2017, Oracle and/or its affiliates. All rights reserved.
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
package jdk.tools.jlink.internal;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.ByteOrder;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.jimage.decompressor.CompressedResourceHeader;
    @Positive
import jdk.internal.module.Resources;
    @Positive
import jdk.internal.module.ModuleInfo;
    @Positive
import jdk.internal.module.ModuleInfo.Attributes;
    @Positive
import jdk.internal.module.ModuleTarget;
    @Positive
import jdk.tools.jlink.plugin.ResourcePool;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolBuilder;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolEntry;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolModule;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolModuleView;
    @Positive
import jdk.tools.jlink.plugin.PluginException;

    @Positive
public class ResourcePoolManager {

    @Positive
    static Attributes readModuleAttributes(ResourcePoolModule mod);

    @Positive
    public static boolean isNamedPackageResource(String path);

    @Positive
    class ResourcePoolModuleImpl implements ResourcePoolModule {

    @Positive
        @Override
    @Positive
        public String name();

    @Positive
        @Override
    @Positive
        public Optional<ResourcePoolEntry> findEntry(String path);

    @Positive
        @Override
    @Positive
        public ModuleDescriptor descriptor();

    @Positive
        @Override
    @Positive
        public String targetPlatform();

    @Positive
        @Override
    @Positive
        public Set<String> packages();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public Stream<ResourcePoolEntry> entries();

    @Positive
        @Override
    @Positive
        public int entryCount();
    @Positive
    }

    @Positive
    public class ResourcePoolImpl implements ResourcePool {

    @Positive
        @Override
    @Positive
        public ResourcePoolModuleView moduleView();

    @Positive
        @Override
    @Positive
        public Stream<ResourcePoolEntry> entries();

    @Positive
        @Override
    @Positive
        public int entryCount();

    @Positive
        @Override
    @Positive
        public Optional<ResourcePoolEntry> findEntry(String path);

    @Positive
        @Override
    @Positive
        public Optional<ResourcePoolEntry> findEntryInContext(String path, ResourcePoolEntry context);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean contains(ResourcePoolEntry data);

    @Positive
        @Override
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        public ByteOrder byteOrder();

    @Positive
        public StringTable getStringTable();
    @Positive
    }

    @Positive
    class ResourcePoolBuilderImpl implements ResourcePoolBuilder {

    @Positive
        @Override
    @Positive
        public void add(ResourcePoolEntry data);

    @Positive
        @Override
    @Positive
        public ResourcePool build();
    @Positive
    }

    @Positive
    class ResourcePoolModuleViewImpl implements ResourcePoolModuleView {

    @Positive
        @Override
    @Positive
        public Optional<ResourcePoolModule> findModule(String name);

    @Positive
        @Override
    @Positive
        public Stream<ResourcePoolModule> modules();

    @Positive
        @Override
    @Positive
        public int moduleCount();
    @Positive
    }

    @Positive
    public ResourcePoolManager() {
    @Positive
    }

    @Positive
    public ResourcePoolManager(ByteOrder order) {
    @Positive
    }

    @Positive
    public ResourcePoolManager(ByteOrder order, StringTable table) {
    @Positive
    }

    @Positive
    public ResourcePool resourcePool();

    @Positive
    public ResourcePoolBuilder resourcePoolBuilder();

    @Positive
    public ResourcePoolModuleView moduleView();

    @Positive
    public void add(ResourcePoolEntry data);

    @Positive
    public Optional<ResourcePoolModule> findModule(String name);

    @Positive
    public Stream<ResourcePoolModule> modules();

    @Positive
    public int moduleCount();

    @Positive
    public Stream<ResourcePoolEntry> entries();

    @Positive
    public int entryCount();

    @Positive
    public Optional<ResourcePoolEntry> findEntry(String path);

    @Positive
    public Optional<ResourcePoolEntry> findEntryInContext(String path, ResourcePoolEntry context);

    @Positive
    @Pure
    @Positive
    public boolean contains(ResourcePoolEntry data);

    @Positive
    public boolean isEmpty();

    @Positive
    public ByteOrder byteOrder();

    @Positive
    public StringTable getStringTable();

    @Positive
    public static final class CompressedModuleData extends ByteArrayResourcePoolEntry {

    @Positive
        public long getUncompressedSize();

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static CompressedModuleData newCompressedResource(ResourcePoolEntry original, ByteBuffer compressed, String plugin, String pluginConfig, StringTable strings, ByteOrder order);
    @Positive
}

// CFWR semantic augmentation - variant 0
