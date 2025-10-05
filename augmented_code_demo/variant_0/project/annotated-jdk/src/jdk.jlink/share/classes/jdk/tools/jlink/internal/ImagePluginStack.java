/*
    @Positive
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.DataOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.nio.ByteOrder;
    @Positive
import java.util.*;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.jimage.decompressor.Decompressor;
    @Positive
import jdk.internal.module.ModuleInfo.Attributes;
    @Positive
import jdk.internal.module.ModuleTarget;
    @Positive
import jdk.tools.jlink.builder.ImageBuilder;
    @Positive
import jdk.tools.jlink.plugin.Plugin;
    @Positive
import jdk.tools.jlink.plugin.PluginException;
    @Positive
import jdk.tools.jlink.plugin.ResourcePool;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolEntry;
    @Positive
import jdk.tools.jlink.plugin.ResourcePoolModule;

    @Positive
public final class ImagePluginStack {

    @Positive
    public interface ImageProvider {

    @Positive
        ExecutableImage retrieve(ImagePluginStack stack) throws IOException;
    @Positive
    }

    @Positive
    public static final class OrderedResourcePoolManager extends ResourcePoolManager {

    @Positive
        class OrderedResourcePool extends ResourcePoolImpl {

    @Positive
            List<ResourcePoolEntry> getOrderedList();
    @Positive
        }

    @Positive
        public OrderedResourcePoolManager(ByteOrder order, StringTable table) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public ResourcePool resourcePool();

    @Positive
        @Override
    @Positive
        public void add(ResourcePoolEntry resource);

    @Positive
        List<ResourcePoolEntry> getOrderedList();
    @Positive
    }

    @Positive
    private final static class CheckOrderResourcePoolManager extends ResourcePoolManager {

    @Positive
        public CheckOrderResourcePoolManager(ByteOrder order, List<ResourcePoolEntry> orderedList, StringTable table) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void add(ResourcePoolEntry resource);
    @Positive
    }

    @Positive
    private static final class PreVisitStrings implements StringTable {

    @Positive
        @Override
    @Positive
        public int addString(String str);

    @Positive
        @Override
    @Positive
        public String getString(int id);
    @Positive
    }

    @Positive
    public ImagePluginStack() {
    @Positive
    }

    @Positive
    public ImagePluginStack(ImageBuilder imageBuilder, List<Plugin> plugins, Plugin lastSorter) {
    @Positive
    }

    @Positive
    public ImagePluginStack(ImageBuilder imageBuilder, List<Plugin> plugins, Plugin lastSorter, boolean validate) {
    @Positive
    }

    @Positive
    public void operate(ImageProvider provider) throws Exception;

    @Positive
    public DataOutputStream getJImageFileOutputStream() throws IOException;

    @Positive
    public ImageBuilder getImageBuilder();

    @Positive
    public ResourcePool visitResources(ResourcePoolManager resources) throws Exception;

    @Positive
    private class LastPoolManager extends ResourcePoolManager {

    @Positive
        private class LastModule implements ResourcePoolModule {

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
        @Override
    @Positive
        public void add(ResourcePoolEntry resource);

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
        public boolean contains(ResourcePoolEntry res);

    @Positive
        @Override
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        public ByteOrder byteOrder();
    @Positive
    }

    @Positive
    public void storeFiles(ResourcePool original, ResourcePool transformed, BasicImageWriter writer) throws Exception;

    @Positive
    public ExecutableImage getExecutableImage() throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
