/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import com.sun.source.doctree.UnknownBlockTagTree;
    @Positive
import jdk.javadoc.doclet.Taglet.Location;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;

    @Positive
public class BaseTaglet implements Taglet {

    @Positive
    protected final DocTree.Kind tagKind;

    @Positive
    protected final String name;

    @Positive
    @Override
    @Positive
    public Set<Location> getAllowedLocations();

    @Positive
    @Override
    @Positive
    public final boolean inField();

    @Positive
    @Override
    @Positive
    public final boolean inConstructor();

    @Positive
    @Override
    @Positive
    public final boolean inMethod();

    @Positive
    @Override
    @Positive
    public final boolean inOverview();

    @Positive
    @Override
    @Positive
    public final boolean inModule();

    @Positive
    @Override
    @Positive
    public final boolean inPackage();

    @Positive
    @Override
    @Positive
    public final boolean inType();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public final boolean isInlineTag();

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    public DocTree.Kind getTagKind();

    @Positive
    public boolean accepts(DocTree tree);

    @Positive
    @Override
    @Positive
    public Content getInlineTagOutput(Element element, DocTree tag, TagletWriter writer);

    @Positive
    @Override
    @Positive
    public Content getAllBlockTagOutput(Element holder, TagletWriter writer);
    @Positive
}
