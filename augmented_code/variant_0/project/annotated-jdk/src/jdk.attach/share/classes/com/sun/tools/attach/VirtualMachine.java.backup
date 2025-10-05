/*
 * Copyright (c) 2005, 2014, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.attach;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.tools.attach.spi.AttachProvider;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.io.IOException;

public abstract class VirtualMachine {

    protected VirtualMachine(AttachProvider provider, String id) {
    }

    public static List<VirtualMachineDescriptor> list();

    public static VirtualMachine attach(String id) throws AttachNotSupportedException, IOException;

    public static VirtualMachine attach(VirtualMachineDescriptor vmd) throws AttachNotSupportedException, IOException;

    public abstract void detach() throws IOException;

    public final AttachProvider provider();

    public final String id();

    public abstract void loadAgentLibrary(String agentLibrary, String options) throws AgentLoadException, AgentInitializationException, IOException;

    public void loadAgentLibrary(String agentLibrary) throws AgentLoadException, AgentInitializationException, IOException;

    public abstract void loadAgentPath(String agentPath, String options) throws AgentLoadException, AgentInitializationException, IOException;

    public void loadAgentPath(String agentPath) throws AgentLoadException, AgentInitializationException, IOException;

    public abstract void loadAgent(String agent, String options) throws AgentLoadException, AgentInitializationException, IOException;

    public void loadAgent(String agent) throws AgentLoadException, AgentInitializationException, IOException;

    public abstract Properties getSystemProperties() throws IOException;

    public abstract Properties getAgentProperties() throws IOException;

    public abstract void startManagementAgent(Properties agentProperties) throws IOException;

    public abstract String startLocalManagementAgent() throws IOException;

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object ob);

    public String toString();
}
