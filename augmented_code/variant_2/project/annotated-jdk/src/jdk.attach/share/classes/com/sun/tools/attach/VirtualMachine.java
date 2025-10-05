/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2014, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package com.sun.tools.attach;

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
import com.sun.tools.attach.spi.AttachProvider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Properties;
    @Positive
import java.io.IOException;

    @Positive
public abstract class VirtualMachine {

    @Positive
    protected VirtualMachine(AttachProvider provider, String id) {
    @Positive
    }

    @Positive
    public static List<VirtualMachineDescriptor> list();

    @Positive
    public static VirtualMachine attach(String id) throws AttachNotSupportedException, IOException;

    @Positive
    public static VirtualMachine attach(VirtualMachineDescriptor vmd) throws AttachNotSupportedException, IOException;

    @Positive
    public abstract void detach() throws IOException;

    @Positive
    public final AttachProvider provider();

    @Positive
    public final String id();

    @Positive
    public abstract void loadAgentLibrary(String agentLibrary, String options) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public void loadAgentLibrary(String agentLibrary) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public abstract void loadAgentPath(String agentPath, String options) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public void loadAgentPath(String agentPath) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public abstract void loadAgent(String agent, String options) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public void loadAgent(String agent) throws AgentLoadException, AgentInitializationException, IOException;

    @Positive
    public abstract Properties getSystemProperties() throws IOException;

    @Positive
    public abstract Properties getAgentProperties() throws IOException;

    @Positive
    public abstract void startManagementAgent(Properties agentProperties) throws IOException;

    @Positive
    public abstract String startLocalManagementAgent() throws IOException;

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    public String toString();
    @Positive
}
